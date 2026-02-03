import subprocess
import numpy as np
import cv2
from rknnlite.api import RKNNLite
import threading
import queue
import time
import json
import io
import os
from minio import Minio
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# load .env config
load_dotenv()

# url / device config
RTMP_URL = os.getenv("RTMP_URL", "rtmp://127.0.0.1/live/test")
VIDEO_DEV = os.getenv("VIDEO_DEV", "/dev/video0")
RKNN_MODEL = os.getenv("RKNN_MODEL", "yolo26n-rk3588.rknn")
PUSH_SIZE = 640

# 推流模式
STREAM_MODE = os.getenv("STREAM_MODE", "raw_1080p")

# minio mqtt config
MINIO_CONF = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "bucket": os.getenv("MINIO_BUCKET", "detections"),
    # convert str
    "secure": os.getenv("MINIO_SECURE", "False").lower() == "true"
}

MQTT_CONF = {
    "host": os.getenv("MQTT_HOST", "localhost"),
    "port": int(os.getenv("MQTT_PORT", 1883)),
    "topic": os.getenv("MQTT_TOPIC", "rk3588/alarms")
}

# logics
UPLOAD_COOLDOWN = float(os.getenv("UPLOAD_COOLDOWN", 5.0))
last_upload_times = {} 

# parse alarm class
alarm_classes_str = os.getenv("ALARM_CLASSES", "person,car")
ALARM_CLASSES = set(alarm_classes_str.split(","))

# coco80 class def
CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush")

# queue
frame_queue = queue.Queue(maxsize=1)
push_queue = queue.Queue(maxsize=2)
upload_queue = queue.Queue(maxsize=5)

# 异步上传 Worker
def upload_worker():
    """负责 MinIO 上传和 MQTT 消息发布的线程"""
    try:
        minio_client = Minio(
            MINIO_CONF["endpoint"],
            access_key=MINIO_CONF["access_key"],
            secret_key=MINIO_CONF["secret_key"],
            secure=MINIO_CONF["secure"]
        )
        
        if not minio_client.bucket_exists(MINIO_CONF["bucket"]):
            minio_client.make_bucket(MINIO_CONF["bucket"])
        
        mqtt_client = mqtt.Client()
        mqtt_client.connect(MQTT_CONF["host"], MQTT_CONF["port"])
        mqtt_client.loop_start()

        print(f">>> 异步上传 Worker 已就绪 (报警目标: {ALARM_CLASSES})")
    except Exception as e:
        print(f">>> Worker 初始化失败: {e}")
        return

    while True:
        task = upload_queue.get()
        if task is None: break
        
        timestamp, frame, alarm_boxes, alarm_labels, alarm_scores = task
        
        success, encoded_img = cv2.imencode('.jpg', frame)
        if not success: continue
        
        img_bytes = encoded_img.tobytes()
        file_name = f"alarm_{timestamp}.jpg"
        
        try:
            # 上传MinIO
            minio_client.put_object(
                MINIO_CONF["bucket"],
                file_name,
                io.BytesIO(img_bytes),
                len(img_bytes),
                content_type='image/jpeg'
            )
            
            #  MQTT消息
            payload = {
                "device_id": "rk3588_node_01",
                "timestamp": timestamp,
                "image_url": f"http{'s' if MINIO_CONF['secure'] else ''}://{MINIO_CONF['endpoint']}/{MINIO_CONF['bucket']}/{file_name}",
                "detections": []
            }
            for b, l, s in zip(alarm_boxes, alarm_labels, alarm_scores):
                payload["detections"].append({
                    "class": CLASSES[l],
                    "score": round(float(s), 2),
                    "box": b
                })
            
            mqtt_client.publish(MQTT_CONF["topic"], json.dumps(payload))
            print(f"[Alarm Triggered] 已上传至 MinIO 并发送 MQTT: {CLASSES[alarm_labels[0]]}")
            
        except Exception as e:
            print(f"[Upload Error] {e}")

# --- 5. 核心组件 ---
def ffmpeg_reader():
    """
    根据 STREAM_MODE 决定 FFmpeg 的工作方式
    """
    if STREAM_MODE == "process_640p":
        # 预处理裁切后给 Python， 一路
        cmd = [
            'ffmpeg', '-f', 'v4l2', '-probesize', '32M', '-video_size', '1920x1080', '-i', VIDEO_DEV,
            '-vf', 'crop=1080:1080:420:0,scale=640:640,format=bgr24', 
            '-f', 'rawvideo', '-r', '30', 'pipe:1'
        ]
    else:
        # 一路推流 1080p (NV12 硬件编码)，一路输出 640p BGR 给模型 (pipe:1)
        cmd = [
            'ffmpeg', '-f', 'v4l2', '-probesize', '32M', '-video_size', '1920x1080', '-i', VIDEO_DEV,
            '-filter_complex', 
            '[0:v]split=2[stream][ai];'
            '[stream]setpts=N/30/TB,format=nv12[s_out];'
            '[ai]crop=1080:1080:420:0,scale=640:640,format=bgr24[ai_out]',
            
            # 输出 1: RTMP 推流 (1080p)
            '-map', '[s_out]', 
            '-c:v', 'h264_rkmpp', '-b:v', '4M', '-g', '30', '-bf', '0', 
            '-f', 'flv', RTMP_URL,
            
            # 输出 2: Python 管道 (640p AI)
            '-map', '[ai_out]', 
            '-f', 'rawvideo', '-r', '30', 'pipe:1'
        ]

    print(f"启动 FFmpeg Reader, 模式: {STREAM_MODE}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)
    
    while True:
        # 无论哪种模式，Python 读到的永远是 AI 用的 640x640 数据
        raw_frame = process.stdout.read(640*640*3)
        if not raw_frame: break
        if frame_queue.full(): frame_queue.get_nowait()
        frame_queue.put(raw_frame)

def ffmpeg_pusher():
    """
    Python 端的推流器。
    仅在 process_640p 模式下工作。raw_1080p 模式下由 Reader 直接推流。
    """
    if STREAM_MODE == "raw_1080p":
        print(">>> 1080p 原始流模式：Python 推流线程已禁用 (由 FFmpeg 直接处理)")
        return

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '640x640', '-r', '30', '-i', 'pipe:0',
        '-c:v', 'h264_rkmpp', '-b:v', '4M', '-g', '30', '-f', 'flv', RTMP_URL
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10**7)
    while True:
        frame = push_queue.get()
        if frame is None: break
        try:
            process.stdin.write(frame.tobytes())
            process.stdin.flush()
        except: 
            break

def post_process_ultralytics(outputs):
    data = outputs[0]
    if len(data.shape) == 3: data = data[0]
    if data.shape[0] < data.shape[1]: data = data.T 
    
    boxes_raw = data[:, :4]
    scores_raw = data[:, 4:]
    class_ids = np.argmax(scores_raw, axis=1)
    confidences = np.max(scores_raw, axis=1)
    
    mask = confidences > 0.45
    if not np.any(mask): return [], [], []
    
    f_conf = confidences[mask]
    f_cls = class_ids[mask]
    f_box_raw = boxes_raw[mask]
    
    boxes = []
    for i in range(len(f_box_raw)):
        cx, cy, w, h = f_box_raw[i]
        boxes.append([int(cx-w/2), int(cy-h/2), int(w), int(h)])
    return boxes, f_conf.tolist(), f_cls.tolist()

# --- 6. 主循环 ---
def main():
    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) != 0: 
        print(f"无法加载模型: {RKNN_MODEL}, 请检查 .env 配置或文件路径")
        return
    rknn.init_runtime(core_mask=0)

    threading.Thread(target=ffmpeg_reader, daemon=True).start()
    threading.Thread(target=ffmpeg_pusher, daemon=True).start()
    threading.Thread(target=upload_worker, daemon=True).start()

    print(f"系统启动：监控中 (模式: {STREAM_MODE})")
    last_fps_time = time.time()
    count = 0

    try:
        while True:
            raw_frame = frame_queue.get()
            draw_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((640,640,3)).copy()
            
            input_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            outputs = rknn.inference(inputs=[np.expand_dims(input_frame, 0)])
            boxes, scores, class_ids = post_process_ultralytics(outputs)
            
            current_alarm_info = [] 
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.45)
                if len(indices) > 0:
                    now = time.time()
                    for i in indices.flatten():
                        cls_name = CLASSES[class_ids[i]].strip()
                        
                        # 过滤器逻辑
                        if cls_name in ALARM_CLASSES:
                            if now - last_upload_times.get(cls_name, 0) > UPLOAD_COOLDOWN:
                                if scores[i] > 0.8:
                                    current_alarm_info.append({
                                        "box": boxes[i],
                                        "label": class_ids[i],
                                        "score": scores[i],
                                        "name": cls_name
                                    })
                                    last_upload_times[cls_name] = now
                        
                        x,y,w,h = boxes[i]
                        cv2.rectangle(draw_frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
                        cv2.putText(draw_frame, cls_name, (x,y-5), 1, 1, (0, 255, 0), 1)

            if STREAM_MODE == "process_640p":
                if push_queue.full(): push_queue.get_nowait()
                push_queue.put(draw_frame)
            
            if len(current_alarm_info) > 0:
                alarm_boxes = [d["box"] for d in current_alarm_info]
                alarm_labels = [d["label"] for d in current_alarm_info]
                alarm_scores = [d["score"] for d in current_alarm_info]
                
                upload_queue.put((int(time.time()), draw_frame.copy(), alarm_boxes, alarm_labels, alarm_scores))

            count += 1
            if count % 30 == 0:
                print(f"FPS: {30/(time.time()-last_fps_time):.2f}")
                last_fps_time = time.time()
                
    except KeyboardInterrupt:
        rknn.release()

if __name__ == '__main__':
    main()
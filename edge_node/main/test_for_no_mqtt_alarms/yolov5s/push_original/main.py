import subprocess
import numpy as np
import cv2
import threading
import queue
import time
import os
from rknnlite.api import RKNNLite
from dotenv import load_dotenv


load_dotenv()

RTMP_URL = os.getenv("RTMP_URL")
VIDEO_DEV = os.getenv("VIDEO_DEV", "/dev/video11")
WIDTH = int(os.getenv("WIDTH", 1920))
HEIGHT = int(os.getenv("HEIGHT", 1080))
RKNN_MODEL = os.getenv("RKNN_MODEL", "yolov5s_relu.rknn")
OBJ_THRESH = float(os.getenv("OBJ_THRESH", 0.25))
NMS_THRESH = float(os.getenv("NMS_THRESH", 0.45))
CORE_MASK = int(os.getenv("CORE_MASK", 1))

CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush")

# 构建 FFmpeg 命令
# split=2 将原始流复制一份用于高清 MPP 编码，另一份缩放给推理
cmd = [
    'ffmpeg', '-f', 'v4l2', '-probesize', '20M', '-video_size', f'{WIDTH}x{HEIGHT}', '-i', VIDEO_DEV,
    '-filter_complex', 
    f'[0:v]split=2[v1][v2]; '
    f'[v1]format=nv12[stream]; '
    f'[v2]fps=30,scale=640:640,format=rgb24[for_python]',
    
    '-map', '[stream]', 
    '-c:v', 'h264_rkmpp', '-b:v', '8M', '-g', '15', '-bf', '0', 
    '-profile:v', 'high', '-level', '4.1', '-r', '60', '-f', 'flv', RTMP_URL,
    
    '-map', '[for_python]', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'
]

# 帧缓冲区逻辑
frame_len = 640 * 640 * 3
frame_queue = queue.Queue(maxsize=1)

def ffmpeg_reader(process):
    while True:
        raw_frame = process.stdout.read(frame_len)
        if len(raw_frame) != frame_len:
            break
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(raw_frame)

def post_process(outputs):
    data = outputs[0][0] 
    boxes, confs, class_ids = [], [], []
    
    mask = data[:, 4] > OBJ_THRESH
    hits = data[mask]
    
    for hit in hits:
        obj_conf = hit[4]
        class_scores = hit[5:]
        class_id = np.argmax(class_scores)
        confidence = obj_conf * class_scores[class_id]
        
        if confidence > OBJ_THRESH:
            cx, cy, w, h = hit[:4] * 640
            x, y = int(cx - w/2), int(cy - h/2)
            
            # 映射回原始分辨率
            real_x = int(x * (WIDTH / 640))
            real_y = int(y * (HEIGHT / 640))
            real_w = int(w * (WIDTH / 640))
            real_h = int(h * (HEIGHT / 640))
            
            boxes.append([real_x, real_y, real_w, real_h])
            confs.append(float(confidence))
            class_ids.append(class_id)
            
    return boxes, confs, class_ids

def main():
    if not RTMP_URL:
        print("错误: 未配置 RTMP_URL")
        return

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)

    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print("模型加载失败！")
        process.terminate()
        return
    
    rknn.init_runtime(core_mask=CORE_MASK)

    t_reader = threading.Thread(target=ffmpeg_reader, args=(process,), daemon=True)
    t_reader.start()

    print(f"系统启动：源({VIDEO_DEV}) -> 推流({RTMP_URL})")
    
    last_time = time.time()
    count = 0

    try:
        while True:
            raw_frame = frame_queue.get()
            frame_img = np.frombuffer(raw_frame, dtype=np.uint8).reshape((640, 640, 3))
            
            input_data = np.expand_dims(frame_img, axis=0)
            outputs = rknn.inference(inputs=[input_data])
            
            boxes, confs, class_ids = post_process(outputs)
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confs, OBJ_THRESH, NMS_THRESH)
                if len(indices) > 0:
                    indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
                    for i in indices:
                        if confs[i] > 0.90: 
                            print(f"[{time.strftime('%H:%M:%S')}] 检测到 {CLASSES[class_ids[i]]}: {confs[i]:.2f} at {boxes[i]}")

            count += 1
            if count % 30 == 0:
                fps = 30 / (time.time() - last_time)
                print(f">>> 推理 FPS: {fps:.2f}")
                last_time = time.time()

    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        process.terminate()
        rknn.release()

if __name__ == '__main__':
    main()
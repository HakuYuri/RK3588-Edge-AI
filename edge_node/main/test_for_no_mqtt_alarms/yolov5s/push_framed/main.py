import subprocess
import numpy as np
import cv2
import threading
import queue
import time
import os
from rknnlite.api import RKNNLite
from dotenv import load_dotenv

# load config
load_dotenv()

# 视频流配置
RTMP_URL = os.getenv("RTMP_URL", "rtmp://127.0.0.1/live/test")
VIDEO_DEV = os.getenv("VIDEO_DEV", "/dev/video11")
WIDTH = int(os.getenv("WIDTH", 1920))
HEIGHT = int(os.getenv("HEIGHT", 1080))
PUSH_WIDTH = int(os.getenv("PUSH_WIDTH", 640))
PUSH_HEIGHT = int(os.getenv("PUSH_HEIGHT", 640))

# 模型配置
RKNN_MODEL = os.getenv("RKNN_MODEL", "yolov5s_relu.rknn")
OBJ_THRESH = float(os.getenv("OBJ_THRESH", 0.25))
NMS_THRESH = float(os.getenv("NMS_THRESH", 0.45))
IMG_SIZE = PUSH_WIDTH # 正方形输入

# 类别定义 
CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush")

MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
           [59, 119], [116, 90], [156, 198], [373, 326]]

# 后处理逻辑
def process_scale(input_data, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = input_data.shape[0:2]
    box_confidence = input_data[..., 4:5]
    box_class_probs = input_data[..., 5:]
    box_xy = input_data[..., :2] * 2 - 0.5
    col = np.tile(np.arange(0, grid_w), grid_h).reshape(grid_h, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
    grid = np.stack((col, row), axis=-1).astype(np.float32)
    grid = np.expand_dims(grid, axis=-2).repeat(3, axis=-2)
    box_xy += grid
    box_xy *= (IMG_SIZE / grid_h)
    box_wh = (input_data[..., 2:4] * 2) ** 2
    box_wh *= anchors
    return np.concatenate((box_xy, box_wh), axis=-1), box_confidence, box_class_probs

def yolov5_post_process(outputs):
    if outputs is None: return [], [], []
    all_boxes, all_confidences, all_class_probs = [], [], []
    for i in range(3):
        data = outputs[i]
        if data.ndim == 4 and data.shape[0] == 1: data = data[0]
        if data.shape[0] == 255: data = data.transpose(1, 2, 0)
        h, w = data.shape[0], data.shape[1]
        data = data.reshape(h, w, 3, 85)
        box, conf, prob = process_scale(data, MASKS[i], ANCHORS)
        all_boxes.append(box.reshape(-1, 4))
        all_confidences.append(conf.reshape(-1))
        all_class_probs.append(prob.reshape(-1, 80))
    boxes, confidences, class_probs = np.concatenate(all_boxes), np.concatenate(all_confidences), np.concatenate(all_class_probs)
    _pos = np.where(confidences >= OBJ_THRESH)
    boxes, confidences, class_probs = boxes[_pos], confidences[_pos], class_probs[_pos]
    if len(boxes) == 0: return [], [], []
    class_ids = np.argmax(class_probs, axis=-1)
    final_scores = confidences * np.max(class_probs, axis=-1)
    _keep = np.where(final_scores >= OBJ_THRESH)
    res_boxes = [[int(b[0]-b[2]/2), int(b[1]-b[3]/2), int(b[2]), int(b[3])] for b in boxes[_keep]]
    return res_boxes, final_scores[_keep].tolist(), class_ids[_keep].tolist()

# multi processing
frame_queue = queue.Queue(maxsize=2)
push_queue = queue.Queue(maxsize=2)

def start_push_process():
    # use h264_rkmpp 
    cmd_out = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', 
        '-s', f'{PUSH_WIDTH}x{PUSH_HEIGHT}', '-r', '30', '-i', 'pipe:0',
        '-c:v', 'h264_rkmpp', '-b:v', '4M', '-g', '30', '-bf', '0', 
        '-f', 'flv', RTMP_URL
    ]
    return subprocess.Popen(cmd_out, stdin=subprocess.PIPE, bufsize=10**7)

def ffmpeg_reader():
    # 动态裁剪和缩放
    cmd_in = [
        'ffmpeg', '-f', 'v4l2', '-video_size', f'{WIDTH}x{HEIGHT}', '-i', VIDEO_DEV,
        '-vf', f'crop={HEIGHT}:{HEIGHT}:(iw-ih)/2:0,scale={PUSH_WIDTH}:{PUSH_HEIGHT},format=bgr24', 
        '-f', 'rawvideo', '-r', '30', 'pipe:1'
    ]
    process_in = subprocess.Popen(cmd_in, stdout=subprocess.PIPE, bufsize=10**7)
    frame_size = PUSH_WIDTH * PUSH_HEIGHT * 3
    while True:
        raw_frame = process_in.stdout.read(frame_size)
        if not raw_frame: break
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put(raw_frame)

def ffmpeg_pusher():
    process_out = start_push_process()
    while True:
        draw_frame = push_queue.get()
        if draw_frame is None: break
        try:
            process_out.stdin.write(draw_frame.tobytes())
            process_out.stdin.flush()
        except:
            process_out.terminate()
            process_out = start_push_process()

def main():
    threading.Thread(target=ffmpeg_reader, daemon=True).start()
    threading.Thread(target=ffmpeg_pusher, daemon=True).start()

    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print(f"模型 {RKNN_MODEL} 加载失败")
        return
    
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)
    print(f"RKNN 推理引擎启动: {RKNN_MODEL}")

    last_time = time.time()
    count = 0

    try:
        while True:
            raw_frame = frame_queue.get()
            draw_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((PUSH_HEIGHT, PUSH_WIDTH, 3)).copy()
            
            img_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(img_rgb, axis=0)
            
            outputs = rknn.inference(inputs=[img_input])
            boxes, scores, class_ids = yolov5_post_process(outputs)
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, OBJ_THRESH, NMS_THRESH)
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        label = f"{CLASSES[class_ids[i]]} {scores[i]:.2f}"
                        cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(draw_frame, label, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if push_queue.full():
                try: push_queue.get_nowait()
                except: pass
            push_queue.put(draw_frame)

            count += 1
            if count % 30 == 0:
                print(f"实时推理 FPS: {30/(time.time()-last_time):.2f}")
                last_time = time.time()

    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    finally:
        rknn.release()

if __name__ == '__main__':
    main()
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

# 基础配置
RTMP_URL = os.getenv("RTMP_URL")
VIDEO_DEV = os.getenv("VIDEO_DEV", "/dev/video11")
RKNN_MODEL = os.getenv("RKNN_MODEL", "test.rknn")

# 尺寸与算法配置
WIDTH = int(os.getenv("WIDTH", 1920))
HEIGHT = int(os.getenv("HEIGHT", 1080))
CROP_STR = os.getenv("CROP_STR", "1080:1080:420:0")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", 640))
OBJ_THRESH = float(os.getenv("OBJ_THRESH", 0.4))
NMS_THRESH = float(os.getenv("NMS_THRESH", 0.45))
CORE_MASK = int(os.getenv("CORE_MASK", 1))

# 推流配置
BITRATE = os.getenv("BITRATE", "4M")
FPS = os.getenv("FPS", "30")

CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]

# FFmpeg：裁切和缩放
cmd_in = [
    'ffmpeg', '-f', 'v4l2', '-probesize', '32M', '-video_size', f'{WIDTH}x{HEIGHT}', '-i', VIDEO_DEV,
    '-vf', f'crop={CROP_STR},scale={INPUT_SIZE}:{INPUT_SIZE},format=bgr24', 
    '-f', 'rawvideo', '-r', FPS, 'pipe:1'
]

def start_push_process():
    cmd_out = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', 
        '-s', f'{INPUT_SIZE}x{INPUT_SIZE}', '-r', FPS, '-i', 'pipe:0',
        '-c:v', 'h264_rkmpp', '-b:v', BITRATE, '-g', '30', '-bf', '0', 
        '-profile:v', 'high', '-level', '4.1', 
        '-flvflags', 'no_duration_filesize', '-f', 'flv', RTMP_URL
    ]
    return subprocess.Popen(cmd_out, stdin=subprocess.PIPE, bufsize=10**7)

frame_size = INPUT_SIZE * INPUT_SIZE * 3
frame_queue = queue.Queue(maxsize=2)

def ffmpeg_reader(process):
    while True:
        try:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size: break
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(raw_frame)
        except Exception: break

def post_process_yolo_vectorized(outputs):
    predictions = outputs[0]
    if len(predictions.shape) == 3:
        predictions = predictions[0]

    scores = predictions[:, 4]
    mask = scores > OBJ_THRESH
    hits = predictions[mask]
    
    if len(hits) == 0: return [], [], []

    hits = np.nan_to_num(hits, nan=0.0, posinf=1e6, neginf=-1e6)
    finite_mask = np.isfinite(hits[:, :5]).all(axis=1)
    hits = hits[finite_mask]
    
    if len(hits) == 0: return [], [], []

    boxes_x1y1 = hits[:, :2]
    boxes_x2y2 = hits[:, 2:4]
    boxes_wh = boxes_x2y2 - boxes_x1y1
    
    final_boxes = np.hstack((boxes_x1y1, boxes_wh)) 
    final_scores = hits[:, 4]
    final_class_ids = hits[:, 5].astype(int)
            
    return final_boxes.tolist(), final_scores.tolist(), final_class_ids.tolist()

def main():
    if not RTMP_URL:
        print("Error: RTMP_URL not set in .env")
        return

    process_in = subprocess.Popen(cmd_in, stdout=subprocess.PIPE, bufsize=10**7)
    process_out = start_push_process()

    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print(f"Failed to load model: {RKNN_MODEL}")
        return
    
    rknn.init_runtime(core_mask=CORE_MASK)
    t_reader = threading.Thread(target=ffmpeg_reader, args=(process_in,), daemon=True)
    t_reader.start()

    print(f"System Ready: Model={RKNN_MODEL}, Resolution={INPUT_SIZE}x{INPUT_SIZE}")
    
    last_time = time.time()
    count = 0

    try:
        while True:
            try:
                raw_frame = frame_queue.get(timeout=2)
            except queue.Empty: continue

            draw_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((INPUT_SIZE, INPUT_SIZE, 3)).copy()
            
            input_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_frame, axis=0)
            
            outputs = rknn.inference(inputs=[input_data])
            boxes, scores, class_ids = post_process_yolo_vectorized(outputs)
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, OBJ_THRESH, NMS_THRESH)
                if len(indices) > 0:
                    indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
                    for i in indices:
                        try:
                            x, y, w, h = [int(v) for v in boxes[i]]
                            cls_id = class_ids[i]
                            label_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                            label = f"{label_name}: {scores[i]:.2f}"
                            
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(draw_frame, label, (x, max(0, y - 5)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        except (OverflowError, ValueError): continue

            try:
                process_out.stdin.write(draw_frame.tobytes())
                process_out.stdin.flush()
            except (BrokenPipeError, IOError):
                process_out.terminate()
                process_out.wait()
                process_out = start_push_process()

            count += 1
            if count % 30 == 0:
                fps = 30 / (time.time() - last_time)
                print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.2f} | Detections: {len(boxes)}")
                last_time = time.time()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        process_in.terminate()
        process_out.terminate()
        rknn.release()

if __name__ == '__main__':
    main()
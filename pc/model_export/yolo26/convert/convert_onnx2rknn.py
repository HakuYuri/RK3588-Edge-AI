import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './best.onnx'
RKNN_MODEL = './test.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True

# 请确保类别顺序与训练时一致
CLASSES = ["person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush"]

class_num = len(CLASSES)
nmsThresh = 0.4
objectThresh = 0.5

INPUT_SIZE = 640 

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

def handleResult(detectResult):
    predBoxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    return predBoxs

def postprocess(out, img_h, img_w, padding, ratio):
    print('postprocess ... ')
    left, top, right, bottom = padding
    
    predictions = out[0]
    if len(predictions.shape) == 3:
        predictions = predictions[0]

    detectResult = []
    
    for i in range(predictions.shape[0]):
        row = predictions[i]
        score = row[4]
        if score < objectThresh:
            continue
            
        x1, y1, x2, y2 = row[0:4]
        classId = int(row[5])

        # 映射回原图坐标
        xmin = (x1 - left) / ratio[0]
        ymin = (y1 - top) / ratio[1]
        xmax = (x2 - left) / ratio[0]
        ymax = (y2 - top) / ratio[1]

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w, xmax)
        ymax = min(img_h, ymax)

        box = DetectBox(classId, score, xmin, ymin, xmax, ymax)
        detectResult.append(box)
    
    print(f'Detected {len(detectResult)} boxes above threshold.')
    return handleResult(detectResult)

def export_rknn_inference(img):
    rknn = RKNN(verbose=False)

    print('--> Config model')
    # rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
    rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], 
            target_platform='rk3588',
            quantized_dtype='asymmetric_quantized-8') # 尝试非对称量化
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    print('done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    print('--> Running model')
    outputs = rknn.inference(inputs=[img], data_format='nhwc')
    rknn.release()
    print('done')

    return outputs

def letterbox(img, new_shape=(1088, 1088), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (left, top, right, bottom)

if __name__ == '__main__':
    print('This is main ...')

    img_path = 'bus.jpg'
    if not os.path.exists(img_path):
        orig_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(orig_img, "NO IMAGE FOUND", (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
    else:
        orig_img = cv2.imread(img_path)
        
    img_h, img_w = orig_img.shape[:2]
    
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    process_img, ratio, padding = letterbox(img_rgb, new_shape=(INPUT_SIZE, INPUT_SIZE), auto=False)
    
    img_input = np.expand_dims(process_img, 0)
    
    outputs = export_rknn_inference(img_input)

    predbox = postprocess(outputs, img_h, img_w, padding, ratio)

    for box in predbox:
        xmin, ymin, xmax, ymax = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{CLASSES[box.classId] if box.classId < len(CLASSES) else box.classId}:{box.score:.2f}"
        cv2.putText(orig_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite('./test_rknn_result.jpg', orig_img)
    print(f'Finished! Found {len(predbox)} objects. Result saved to test_rknn_result.jpg')
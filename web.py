# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import csv
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import numpy as np

@torch.no_grad()
def init(
    weights="/data/wcx/code/Global_Wheat_Detection/runs/train/yolov5xse_epoch100_1024/weights/best.pt",  # model.pt path(s)
    imgsz=[1024,1024],  # inference size (pixels)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load model
    device = select_device(device)

    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = (
        model.stride,
        model.names,
        model.pt,
        model.jit,
        model.onnx,
        model.engine,
    )
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (
        pt or jit or engine
    ) and device.type != "cpu"  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

    return model


@torch.no_grad()
def predict_model(
    model,
    image, # (3, 1024, 1024)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    max_det=1000,  # maximum detections per image

):
    device = select_device(device)
    image_numpy = image.copy()
    image = torch.from_numpy(image).to(device)
    image= image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0
    if len(image.shape) == 3:
        image = image[None]  # expand for batch dim

    pred = model(image, augment=False, visualize=False)

    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
    )

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0 = np.ascontiguousarray(np.uint8(image_numpy.copy().transpose((1,2,0))))
        annotator = Annotator(im0, line_width=3, example=str("test"))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                c = int(cls)  # integer class
                label = (None)
                annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
    im0 = annotator.result()
    # cv2.imwrite("result.jpg", im0)
    return im0

if __name__ == "__main__":
    import numpy as np
    model = init(device = "3")
    image = cv2.imread("0002.jpg")
    image = np.transpose(image,(2,0,1))
    im0 = predict_model(model,image,device="3")
    print(im0)

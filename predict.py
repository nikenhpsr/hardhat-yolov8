from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('D:/2023/Coding/project/yolo/yolo/dataset/runs/detect/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model('D:/2023/Coding/project/yolo/yolo/dataset/D09_20230825094234.mp4')  # predict on an image

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk

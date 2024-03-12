from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('D:/2023/Coding/project/yolo/yolo/dataset/runs/detect/train/weights/best.pt')  # load a custom model
source = 'D:/2023/Coding/project/yolo/yolo/dataset/D09_20230825094234.mp4'

cap = cv2.VideoCapture(source)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
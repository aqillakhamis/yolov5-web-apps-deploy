import torch
import cv2
from flask import Flask, Response, render_template
import numpy as np

app = Flask(__name__)

# Load YOLOv5 model using torch.hub
model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s')  # Load YOLOv5 small model

# Initialize webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Helper function to process and return detections on a frame
def detect_objects(frame):
    results = model(frame)  # Perform YOLOv5 inference
    
    # Get predictions (bounding boxes, labels, and confidence scores)
    pred_boxes = results.xywh[0].cpu().numpy()  # Bounding boxes (center_x, center_y, width, height)
    pred_classes = results.pred[0][:, -1].cpu().numpy()  # Class indices
    pred_scores = results.pred[0][:, 4].cpu().numpy()  # Confidence scores
    labels = results.names  # Class names

    # Draw bounding boxes and labels on the frame
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box[:4]
        label = int(pred_classes[i])
        score = pred_scores[i]
        name = labels[label]
        
        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {score:.2f}", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Helper function to generate video stream
def generate():
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process the frame and get detections
        frame = detect_objects(frame)

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame for streaming to the frontend
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    # Use render_template to correctly render the HTML template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the MJPEG stream
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

import cv2
from ultralytics import YOLO

# Load YOLOv8 model (fine-tune or use a pretrained model)
model = YOLO("yolov8n.pt")  # Replace with your custom model if needed

def detect_individuals(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize a list to store detection results
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection on the frame
        results = model(frame)

        frame_detections = []
        for r in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = r
            
            # Filter detections to include only humans
            if model.names[int(cls)] == "person":
                label = f"person {conf:.2f}"

                # Save detection to the list
                frame_detections.append({
                    "class": "person",
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Append frame detections to the main list
        detections.append(frame_detections)

        # Show the frame
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return all detections
    return detections


# Provide path to your video file
video_path = r"C:\Users\vinod\OneDrive\文档\project_drone\vid1.mp4"
output_detections = detect_individuals(video_path)

# Print output detections for the entire video
for frame_index, frame_data in enumerate(output_detections):
    print(f"Frame {frame_index + 1}:")
    for detection in frame_data:
        print(f"  Class: {detection['class']}, Confidence: {detection['confidence']:.2f}, BBox: {detection['bbox']}")

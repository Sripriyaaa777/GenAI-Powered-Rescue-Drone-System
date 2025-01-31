import os
import cv2
from PIL import Image
from ultralytics import YOLO

input_folder = r"C:\Users\Gouri K\Desktop\hyd\GenAI-Powered-Rescue-Drone-System\person_detection\frames2"
output_folder = r"C:\Users\Gouri K\Desktop\hyd\GenAI-Powered-Rescue-Drone-System\person_detection\pframes2"
video_output_path = r"C:\Users\Gouri K\Desktop\hyd\GenAI-Powered-Rescue-Drone-System\person_detection\output_video2.mp4" 

if not os.path.exists(input_folder):
    print(f"The input folder does not exist: {input_folder}")
else:
    print(f"Found input folder: {input_folder}")

model = YOLO("yolov8n.pt") 


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of frames for video creation
frame_list = []

# Process each frame
for frame_file in os.listdir(input_folder):
    frame_path = os.path.join(input_folder, frame_file)
    
    # Ensure it's an image file
    if frame_file.endswith((".jpg", ".png", ".jpeg")):
        # Detect objects
        results = model(frame_path)
        
        for result in results:
            # Filter only 'person' class (class ID 0)
            person_boxes = result.boxes.data[
                result.boxes.data[:, 5] == 0  # Column 5 contains class IDs
            ]
            
            # Replace boxes with filtered person boxes
            result.boxes.data = person_boxes
            
            # Generate the annotated image
            annotated_image = result.plot()
            
            # Save the annotated image
            output_file_path = os.path.join(output_folder, frame_file)
            Image.fromarray(annotated_image).save(output_file_path)

            # Append to frame list for video creation
            frame_list.append(annotated_image)

print("Object detection completed. Results saved in:", output_folder)

# Create video from frames
if frame_list:
    # Get the width and height from the first frame
    height, width, _ = frame_list[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 format
    out = cv2.VideoWriter(video_output_path, fourcc, 30.0, (width, height))  # 30 FPS

    # Write frames to the video
    for frame in frame_list:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

print(f"Video saved at {video_output_path}")

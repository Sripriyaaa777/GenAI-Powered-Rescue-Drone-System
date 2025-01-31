import cv2
import os

# Input and output paths
video_path = r"C:\Users\Gouri K\Desktop\hyd\GenAI-Powered-Rescue-Drone-System\person_detection\vid 2.mp4"
 # Replace with the path to your drone video
output_folder = "frames2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Save each frame as an image
    frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames to the folder '{output_folder}'")

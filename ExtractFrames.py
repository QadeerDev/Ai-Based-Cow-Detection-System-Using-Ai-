import cv2
import os

# Path to input video
video_path = "input\i4.mp4"

# Output folder for frames
output_folder = "frames"

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()

print(f"Done! Extracted {frame_count} frames.")
import cv2
import os

def extract_frames(video_path, output_dir, frame_interval):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Video FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f}s")

    timestamps = []
    frame_id = 0
    next_frame = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        success, frame = cap.read()
        if not success:
            break

        timestamp = next_frame / fps
        filename = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(filename, frame)
        timestamps.append((timestamp, filename))

        frame_id += 1
        next_frame += int(fps * frame_interval)

    cap.release()
    return timestamps

# FIXED file path
video_path = r"video path"
output_dir = "extracted_frames"
frame_interval = 1

timestamps = extract_frames(video_path, output_dir, frame_interval)

for ts, path in timestamps[:5]:
    print(f"Time: {ts:.2f}s -> {path}")

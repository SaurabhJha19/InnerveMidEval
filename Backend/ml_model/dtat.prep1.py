import os
import cv2
import numpy as np


def prepare_deepfake_dataset(real_dir, fake_dir):
    X, y = [], []

    # Process real videos
    for video_file in os.listdir(real_dir):
        video_path = os.path.join(real_dir, video_file)
        frames = extract_frames(video_path)
        X.extend(frames)
        y.extend([1] * len(frames))  # 1 for real

    # Process fake videos
    for video_file in os.listdir(fake_dir):
        video_path = os.path.join(fake_dir, video_file)
        frames = extract_frames(video_path)
        X.extend(frames)
        y.extend([0] * len(frames))  # 0 for fake

    return np.array(X), np.array(y)


def extract_frames(video_path, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and normalize
        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    return frames
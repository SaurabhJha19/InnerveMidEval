import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, List
import cv2
from sklearn.model_selection import train_test_split
import os


class VideoDataPreparator:
    def __init__(self, frame_size: Tuple[int, int], frames_per_video: int):
        self.frame_size = frame_size
        self.frames_per_video = frames_per_video

    def extract_frames(self, video_path: str) -> np.ndarray:
        """Extract frames from a video file"""
        frames = []
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        # Calculate frame indices to extract
        indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if len(frames) != self.frames_per_video:
            return None

        return np.array(frames)

    def prepare_video_dataset(self, real_dir: str, fake_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset from real and fake video directories"""
        X = []
        y = []

        # Process real videos
        print("Processing real videos...")
        for video_path in Path(real_dir).glob('*'):
            if video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                frames = self.extract_frames(str(video_path))
                if frames is not None:
                    X.append(frames)
                    y.append(0)  # Label for real videos

        # Process fake videos
        print("Processing fake videos...")
        for video_path in Path(fake_dir).glob('*'):
            if video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                frames = self.extract_frames(str(video_path))
                if frames is not None:
                    X.append(frames)
                    y.append(1)  # Label for fake videos

        return np.array(X), np.array(y)

    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split dataset into train, validation, and test sets"""
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Second split: 75% train, 25% val (from the remaining 80%)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_frame_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
        """Create TensorFlow dataset with proper batching"""
        # Normalize pixel values
        X = X.astype('float32') / 255.0

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def create_deepfake_detector(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Create the deepfake detector model"""
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


def train_video_deepfake_model(
        real_dir: str,
        fake_dir: str,
        epochs: int = 50,
        batch_size: int = 32,
        frame_size: Tuple[int, int] = (128, 128),
        frames_per_video: int = 20
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train the deepfake detection model using video data"""

    # Initialize data preparator
    data_prep = VideoDataPreparator(
        frame_size=frame_size,
        frames_per_video=frames_per_video
    )

    # Prepare data
    print("Preparing dataset...")
    X, y = data_prep.prepare_video_dataset(real_dir, fake_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.split_dataset(X, y)

    # Create data batches
    train_dataset = data_prep.create_frame_batches(X_train, y_train, batch_size)
    val_dataset = data_prep.create_frame_batches(X_val, y_val, batch_size)
    test_dataset = data_prep.create_frame_batches(X_test, y_test, batch_size)

    # Create and compile model
    model = create_deepfake_detector(input_shape=(*frame_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    # Create model checkpoint directory
    checkpoint_dir = Path('models')
    checkpoint_dir.mkdir(exist_ok=True)

    # Train model
    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'models/deepfake_detector_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3
            )
        ]
    )

    # Evaluate model
    print("Evaluating model...")
    test_results = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")

    return model, history


if __name__ == "__main__":
    # Example usage
    model, history = train_video_deepfake_model(
        real_dir='dataset/real_videos',
        fake_dir='dataset/fake_videos',
        epochs=50,
        batch_size=32
    )

    # Save the final model
    model.save('models/deepfake_detector_final.h5')
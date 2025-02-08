from .model import create_deepfake_model
from .dtat.prep1     import prepare_deepfake_dataset


def train_deepfake_model():
    X, y = prepare_deepfake_dataset(
        real_dir='dataset/real/',
        fake_dir='dataset/fake/'
    )

    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train model
    model = create_deepfake_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32
    )

    # Save model
    model.save('models/deepfake_detector.h5')
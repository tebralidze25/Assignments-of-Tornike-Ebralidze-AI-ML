# train.py - trains a small CNN on images created by generate_images.py
import numpy as np
from pathlib import Path

def main():
    if not Path("dataset_images.npz").exists():
        print("dataset_images.npz not found. Run: python generate_images.py --csv Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        return
    data = np.load("dataset_images.npz", allow_pickle=True)
    filenames = data['filenames']
    labels = data['labels']
    class_names = data['class_names']

    try:
        import imageio
        from sklearn.model_selection import train_test_split
        from tensorflow.keras import layers, models
        from tensorflow.keras.utils import to_categorical
    except Exception as e:
        print("Missing training dependencies. Install tensorflow, imageio, scikit-learn. Error:", e)
        return

    X = []
    for f in filenames:
        img = imageio.v2.imread(f)
        X.append(img)
    X = np.array(X).astype('float32') / 255.0
    # Add a channel dimension for grayscale (32x32x1)
    X = np.expand_dims(X, axis=-1)
    y = to_categorical(labels, num_classes=len(class_names))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = models.Sequential([
        layers.Input(shape=X_train.shape[1:]),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    model.save("ddos_cnn_model.keras")
    print("✅ Training complete. Model saved to ddos_cnn_model.keras")

if __name__ == '__main__':
    main()
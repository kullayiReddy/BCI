import os
import numpy as np
from scipy import signal
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
FS = 250          # sampling rate (Hz) – change if your data is different
SEG_LEN = 512     # samples per window (same as live BUFFER_SIZE)
NPERSEG = 64
NOVERLAP = 32
IMG_SIZE = (128, 128)

CSV_FILES = {
    "Happy": "Dataset/Happy.csv",
    "Tired": "Dataset/Tired.csv",
    "Pain" : "Dataset/Pain.csv"
}

# -----------------------
# 1. Spectrogram function (must match live_detection.py)
# -----------------------
def make_spectrogram(seg):
    """
    seg: 1D numpy array of length SEG_LEN
    returns: (128,128,3) float32 in [0,1]
    """
    f, t, Sxx = signal.spectrogram(
        seg, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    Sxx = np.log(Sxx + 1e-10)
    Sxx -= Sxx.min()
    if Sxx.max() > 0:
        Sxx /= Sxx.max()
    # resize to IMG_SIZE
    import cv2
    Sxx_img = cv2.resize(Sxx, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    Sxx_img = np.stack([Sxx_img]*3, axis=-1)  # 3 channels
    return Sxx_img.astype("float32")

# -----------------------
# 2. Segmenting helper
# -----------------------
def make_segments_from_ts(ts, seg_len=SEG_LEN, hop=None):
    """
    ts: 1D time series
    seg_len: samples per segment
    hop: step size between segments (default seg_len => non-overlapping)
    """
    if hop is None:
        hop = seg_len
    segments = []
    n = len(ts)
    for start in range(0, n - seg_len + 1, hop):
        segments.append(ts[start:start+seg_len])
    return np.array(segments, dtype="float32")

# -----------------------
# 3. Load CSVs -> segments -> spectrograms
# -----------------------
X_images = []
y_labels = []

for label, fname in CSV_FILES.items():
    if not os.path.exists(fname):
        print(f"[WARNING] File not found: {fname}, skipping")
        continue

    print(f"[INFO] Loading {fname} for class '{label}'")
    data = np.loadtxt(fname, delimiter=",")
    
    # Handle different shapes:
    # Case A: 1D -> one long continuous signal
    # Case B: 2D -> many rows, each row = one sample/segment
    if data.ndim == 1:
        print(f"[INFO] {fname} is 1D signal of length {len(data)}")
        segments = make_segments_from_ts(data, seg_len=SEG_LEN, hop=SEG_LEN//2)  # 50% overlap
    else:
        print(f"[INFO] {fname} shape: {data.shape}")
        # data.shape = (num_samples, num_points)
        # Ensure each row has at least SEG_LEN points
        if data.shape[1] < SEG_LEN:
            print(f"[WARNING] rows in {fname} shorter than {SEG_LEN}, skipping file")
            continue
        # trim or take first SEG_LEN from each row
        segments = data[:, :SEG_LEN].astype("float32")

    print(f"[INFO] Got {segments.shape[0]} segments for {label}")

    # Convert each segment to spectrogram image
    for seg in segments:
        img = make_spectrogram(seg)
        X_images.append(img)
        y_labels.append(label)

X_images = np.array(X_images, dtype="float32")  # shape: (N,128,128,3)
y_labels = np.array(y_labels)

print("[INFO] Final dataset shapes:", X_images.shape, y_labels.shape)

# -----------------------
# 4. Encode labels
# -----------------------
le = LabelEncoder()
y_enc = le.fit_transform(y_labels)  # e.g. Happy->0, Pain->1, Tired->2
num_classes = len(le.classes_)
print("[INFO] Label mapping:", dict(zip(le.classes_, range(num_classes))))

# to one-hot
y_onehot = tf.keras.utils.to_categorical(y_enc, num_classes=num_classes)

# -----------------------
# 5. Train/Val split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_images, y_onehot, test_size=0.2, random_state=42, stratify=y_enc
)

print("[INFO] Train:", X_train.shape, "Val:", X_val.shape)

# -----------------------
# 6. Class weights (handle imbalance)
# -----------------------
class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)
class_weight = {i: w for i, w in enumerate(class_weight_values)}
print("[INFO] Class weights:", class_weight)

# -----------------------
# 7. Build CNN model
# -----------------------
def build_model(input_shape=(128,128,3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model

model = build_model(input_shape=(128,128,3), num_classes=num_classes)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# 8. Train
# -----------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    class_weight=class_weight,
    verbose=1
)

# -----------------------
# 9. Evaluate
# -----------------------
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"[INFO] Validation loss: {val_loss:.4f}, acc: {val_acc:.4f}")

# -----------------------
# 10. Save model
# -----------------------
model.save("BCI_emotion_model.h5")
print("[INFO] Saved model as BCI_emotion_model.h5")

# -----------------------
# 11. Plot training curves (optional)
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss")
plt.tight_layout()
plt.show()

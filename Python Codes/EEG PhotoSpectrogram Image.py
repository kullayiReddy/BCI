import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

INPUT_DATASET = "dataset"                  # your raw EEG photo dataset
OUTPUT_DATASET = "spectrogram_dataset"     # folder to save spectrograms
IMG_SIZE = (128, 128)
os.makedirs(OUTPUT_DATASET, exist_ok=True)

def make_spectrogram_from_image(img_path, save_path):
    # Read EEG waveform photo (grayscale)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    # Collapse vertically → 1D signal
    signal_1d = img.mean(axis=0)

    # Compute spectrogram
    f, t, Sxx = spectrogram(signal_1d, fs=250, nperseg=64, noverlap=32)

    # Normalize to 0–255
    Sxx = np.log(Sxx + 1e-7)
    Sxx = cv2.resize(Sxx, IMG_SIZE)
    Sxx = cv2.normalize(Sxx, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to RGB heatmap
    Sxx_rgb = cv2.applyColorMap(Sxx.astype(np.uint8), cv2.COLORMAP_JET)

    # Save
    cv2.imwrite(save_path, Sxx_rgb)

# ---------------------------
# Convert ENTIRE dataset
# ---------------------------
for label in os.listdir(INPUT_DATASET):
    in_label_dir = os.path.join(INPUT_DATASET, label)
    out_label_dir = os.path.join(OUTPUT_DATASET, label)
    os.makedirs(out_label_dir, exist_ok=True)

    for filename in os.listdir(in_label_dir):
        in_path = os.path.join(in_label_dir, filename)
        out_path = os.path.join(out_label_dir, filename.replace(".jpg", "_spec.png"))

        try:
            make_spectrogram_from_image(in_path, out_path)
            print("Converted:", out_path)
        except Exception as e:
            print("Error:", in_path, e)

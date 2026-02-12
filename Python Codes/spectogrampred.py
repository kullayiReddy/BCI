"""
Live EEG Emotion Classification with Signal + Spectrogram Visualization
Classes: happy, pain, tired
ASCII-SAFE version (no emojis, no Unicode)
"""

import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Warning: pyserial not installed. Install with: pip install pyserial")
    serial = None

from scipy import signal
import cv2
import tensorflow as tf

# Try TTS safely
try:
    import pyttsx3
    engine = pyttsx3.init()
except Exception:
    print("Warning: TTS engine failed to initialize.")
    engine = None


# -------------------------
# USER CONFIG
# -------------------------
MODEL_PATH = "EEG_spectrogram_model.h5"   # Your trained model
SAMPLING_RATE = 250
BUFFER_SIZE = 512
STEP_SAMPLES = 64

N_PER_SEG = 64
NO_OVERLAP = 32
IMG_SIZE = (128, 128)

COM_PORT = None       # Auto-detect
BAUDRATE = 115200

NOTCH_FREQ = 50
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0

CLASS_NAMES = ["happy", "pain", "tired"]
CONFIDENCE_THRESHOLD = 0.60
MIN_SECONDS_BETWEEN_SPEECH = 2.0


# -------------------------
# Helper: Serial Port
# -------------------------
def find_arduino_port():
    if COM_PORT:
        return COM_PORT
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return ports[0].device if ports else None


# -------------------------
# DSP Filters
# -------------------------
def design_filters():
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ / (SAMPLING_RATE / 2), 30)
    b_band, a_band = signal.butter(
        4,
        [BANDPASS_LOW / (SAMPLING_RATE / 2), BANDPASS_HIGH / (SAMPLING_RATE / 2)],
        btype="band"
    )
    return (b_notch, a_notch), (b_band, a_band)


def apply_filters(x, notch, band):
    b_notch, a_notch = notch
    b_band, a_band = band
    x = signal.filtfilt(b_notch, a_notch, x)
    x = signal.filtfilt(b_band, a_band, x)
    return x


# -------------------------
# Spectrogram Generation
# -------------------------
def signal_to_spectrogram_frame(sig_1d):
    f, t, Sxx = signal.spectrogram(
        sig_1d,
        fs=SAMPLING_RATE,
        nperseg=N_PER_SEG,
        noverlap=NO_OVERLAP,
    )

    Sxx = np.log(Sxx + 1e-7)
    Sxx = cv2.resize(Sxx, IMG_SIZE)

    Sxx_norm = cv2.normalize(Sxx, None, 0, 255, cv2.NORM_MINMAX)
    Sxx_rgb = cv2.applyColorMap(Sxx_norm.astype(np.uint8), cv2.COLORMAP_JET)
    Sxx_rgb = cv2.cvtColor(Sxx_rgb, cv2.COLOR_BGR2RGB)

    return Sxx_rgb.astype("float32") / 255.0, Sxx_norm


# -------------------------
# TTS Wrapper
# -------------------------
def speak(text):
    if engine is None:
        print("[TTS Disabled] Output:", text)
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        print("TTS Error")


# -------------------------
# MAIN PROGRAM
# -------------------------
def main():

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    notch_filter, bandpass_filter = design_filters()

    # Connect to Arduino
    port = find_arduino_port()
    if port is None:
        print("No Arduino COM port found.")
        return

    print("Connecting to Arduino on", port)
    ser = serial.Serial(port, BAUDRATE, timeout=1)
    time.sleep(2)
    print("Connection established.")

    buffer = deque(maxlen=BUFFER_SIZE)
    sample_count = 0
    last_spoken_label = None
    last_spoken_time = 0

    # -------------------------
    # Create Live Plot Windows
    # -------------------------
    plt.ion()
    fig, (ax_raw, ax_spec) = plt.subplots(2, 1, figsize=(8, 6))

    raw_line, = ax_raw.plot(np.zeros(BUFFER_SIZE))
    ax_raw.set_title("Live EEG Signal")
    ax_raw.set_ylim(0, 1023)

    img_display = ax_spec.imshow(np.zeros((128, 128)), cmap="jet")
    ax_spec.set_title("Live Spectrogram")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # LIVE LOOP
    # -------------------------
    try:
        while True:

            # Read Arduino line
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            # Convert to int safely
            try:
                val = int(line)
            except:
                continue

            buffer.append(val)
            sample_count += 1

            # Need full window
            if len(buffer) < BUFFER_SIZE:
                continue

            # Update raw plot
            raw_line.set_ydata(buffer)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Skip until it's time for prediction
            if sample_count % STEP_SAMPLES != 0:
                continue

            # Convert to numpy
            sig = np.array(buffer, dtype=np.float32)
            sig -= np.mean(sig)
            sig = apply_filters(sig, notch_filter, bandpass_filter)

            # Create spectrogram
            img, spec_gray = signal_to_spectrogram_frame(sig)

            # Update spectrogram plot
            img_display.set_data(spec_gray)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Predict
            inp = np.expand_dims(img, axis=0)
            preds = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            label = CLASS_NAMES[idx]

            print("Predicted:", label, "| Confidence:", round(conf, 2))

            # Speak result
            now = time.time()
            if conf >= CONFIDENCE_THRESHOLD:
                print("Speaking:", label)
                speak(label)
                last_spoken_label = label
                last_spoken_time = now

    except KeyboardInterrupt:
        print("Stopping live detection...")

    finally:
        ser.close()
        print("Serial connection closed.")


# Run program
if __name__ == "__main__":
    main()

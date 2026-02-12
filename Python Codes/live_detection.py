"""
live_detection.py
Hardware : Arduino Nano R3, 1 EEG channel on A0
Sampling : 250 Hz
Model    : spectrogram-based CNN, input 128x128x3
Classes  : Happy, Tired, Pain
"""

import time
from collections import deque
import numpy as np
import serial
import serial.tools.list_ports
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
import threading
import pyttsx3

# -------------------------
# USER CONFIG
# -------------------------
MODEL_PATH = "EEG_spectrogram_model.h5"
COM_PORT   = "COM13"
BAUDRATE   = 115200

SAMPLING_RATE = 250
BUFFER_SIZE   = 512

IMG_SIZE   = (128, 128)
NPERSEG    = 64
NOVERLAP   = 32

BANDPASS_LOW  = 0.5
BANDPASS_HIGH = 40.0
NOTCH_FREQ    = 50

CLASS_LABELS  = ["Happy", "Tired", "Pain"]
NEUTRAL_LABEL = "Neutral"

# Lower threshold to avoid always Neutral
CONF_THRESHOLD = 0.4

STD_MIN  = 3.0
STD_MAX  = 500.0
MEAN_MIN = 0.0
MEAN_MAX = 4096.0

ALPHA = 0.7
MIN_STABLE_COUNT = 3

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# -------------------------
# GLOBALS
# -------------------------
running = True
prev_prediction = None
stable_label    = None
stable_count    = 0
last_spoken_label = None

# -------------------------
# Thread-safe speech
# -------------------------
def speak_text(text: str):
    """Speak text in a background thread."""
    def _run():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logging.warning(f"TTS error: {e}")
    threading.Thread(target=_run, daemon=True).start()

# -------------------------
# Serial helpers
# -------------------------
def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

def open_serial(port=None, baud=115200, timeout=1.0):
    if port is None:
        ports = list_serial_ports()
        logging.info("Available ports: %s", ports)
        if not ports:
            raise RuntimeError("No serial ports found")
        port = ports[-1]

    ser = serial.Serial(port, baud, timeout=timeout)
    time.sleep(1.0)
    ser.reset_input_buffer()
    logging.info("Opened serial %s @%d", port, baud)
    return ser

# -------------------------
# Signal helpers
# -------------------------
def bandpass_filter(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.lfilter(b, a, data)

def notch_filter(data, freq, fs, q=30):
    w0 = freq / (fs/2)
    b, a = signal.iirnotch(w0, q)
    return signal.lfilter(b, a, data)

def is_signal_valid(segment):
    std_val  = np.std(segment)
    mean_val = np.mean(segment)

    if np.isnan(std_val) or np.isnan(mean_val):
        return False, "NaN"
    if std_val < STD_MIN:
        return False, f"Low std {std_val:.2f}"
    if std_val > STD_MAX:
        return False, f"High std {std_val:.2f}"
    if not (MEAN_MIN <= mean_val <= MEAN_MAX):
        return False, f"Mean {mean_val:.1f} out of range"

    return True, f"OK (std={std_val:.2f}, mean={mean_val:.1f})"

def compute_spectrogram_img(data):
    f, t, Sxx = signal.spectrogram(
        data, fs=SAMPLING_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    Sxx = np.log(Sxx + 1e-10)
    Sxx -= Sxx.min()
    if Sxx.max() > 0:
        Sxx /= Sxx.max()
    img = cv2.resize(Sxx, IMG_SIZE)
    img = np.stack([img]*3, axis=-1)
    return img.astype(np.float32)

# -------------------------
# Load model
# -------------------------
if not os.path.exists(MODEL_PATH):
    logging.error("Model not found at %s", MODEL_PATH)
    raise SystemExit(1)

logging.info("Loading model: %s", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
logging.info("Model loaded.")

# -------------------------
# Plot setup
# -------------------------
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6))

line_eeg, = ax1.plot(np.zeros(SAMPLING_RATE))
ax1.set_ylim(0, 4096)
ax1.set_xlim(0, SAMPLING_RATE)
ax1.set_title("Live EEG (1 second window)")

img_display = ax2.imshow(
    np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3)),
    origin='lower',
    aspect='auto'
)
ax2.set_title("Spectrogram")
plt.tight_layout()
plt.show(block=False)

# -------------------------
# Key press handler
# -------------------------
def on_key(event):
    global running
    if event.key in ("q", "Q"):
        logging.info("Q pressed – exiting...")
        running = False

fig.canvas.mpl_connect('key_press_event', on_key)

# -------------------------
# Serial open
# -------------------------
try:
    ser = open_serial(port=COM_PORT, baud=BAUDRATE)
except Exception as e:
    logging.error(f"Failed to open serial port {COM_PORT}: {e}")
    raise SystemExit(1)

buffer = deque(maxlen=BUFFER_SIZE)

log_file = open("prediction_log.csv", "a")
log_file.write("time,label,confidence,info\n")

logging.info("Live detection started. Press Q in plot window to stop.")

# -------------------------
# Main loop
# -------------------------
try:
    while running:

        # Process matplotlib events to handle key presses
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue

            val = int(float(line))
        except Exception as e:
            logging.debug(f"Serial read error: {e}")
            continue

        buffer.append(val)

        # update EEG plot
        if len(buffer) >= SAMPLING_RATE:
            last_1s = list(buffer)[-SAMPLING_RATE:]
            line_eeg.set_ydata(last_1s)
            line_eeg.set_xdata(np.arange(len(last_1s)))
            ax1.relim()
            ax1.autoscale_view()

        if len(buffer) < BUFFER_SIZE:
            continue

        segment = np.array(buffer, dtype=np.float32)
        timestamp = time.time()

        # filtering
        try:
            segment_f = bandpass_filter(segment, BANDPASS_LOW, BANDPASS_HIGH, SAMPLING_RATE)
            segment_f = notch_filter(segment_f, NOTCH_FREQ, SAMPLING_RATE)
        except:
            segment_f = segment

        valid, info = is_signal_valid(segment_f)
        if not valid:
            logging.info("NO SIGNAL: " + info)
            img_display.set_data(np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3)))
            plt.pause(0.001)
            buffer.clear()
            continue

        img = compute_spectrogram_img(segment_f)
        img_display.set_data(img)
        plt.pause(0.001)

        x = np.expand_dims(img, axis=0)
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = CLASS_LABELS[idx]

        logging.info(f"RAW PRED → {label} (conf={conf:.3f})")

        # confidence threshold
        if conf < CONF_THRESHOLD:
            raw_label = NEUTRAL_LABEL
        else:
            raw_label = label

        # smoothing
        if raw_label == stable_label:
            stable_count += 1
        else:
            stable_label = raw_label
            stable_count = 1

        if stable_count >= MIN_STABLE_COUNT:
            final_label = stable_label
        else:
            final_label = NEUTRAL_LABEL

        logging.info(f"FINAL LABEL → {final_label} (stable_count={stable_count})")

        # speak if changed
        if final_label != NEUTRAL_LABEL and final_label != last_spoken_label:
            speak_text(final_label)
            last_spoken_label = final_label

        log_file.write(f"{timestamp},{final_label},{conf:.3f},{info}\n")
        buffer.clear()

except KeyboardInterrupt:
    running = False

finally:
    ser.close()
    log_file.close()
    plt.close('all')
    logging.info("Exited cleanly.")

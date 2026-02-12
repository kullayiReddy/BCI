"""
live_detection_fixed.py
Stable plotting + repeated voice + queue-based TTS
"""

import time
import sys
import threading
from collections import deque
import queue
import numpy as np
import serial
import serial.tools.list_ports
from scipy import signal
import cv2
import tensorflow as tf
import os
import pyttsx3
import logging
import matplotlib.pyplot as plt

# -------------------------
# USER CONFIG
# -------------------------
MODEL_PATH = "EEG_spectrogram_model.h5"

SAMPLING_RATE = 250
BUFFER_SIZE = 512

N_PER_SEG = 64
N_OVERLAP = 32
IMG_SIZE = (128,128)

COM_PORT = "COM13"
BAUDRATE = 115200

NOTCH_FREQ = 50
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0

CLASS_LABELS = ["Happy", "Tired", "Pain", "Neutral"]
NEUTRAL_LABEL = "Neutral"
CONF_THRESHOLD = 0.6

STD_MIN = 3.0
STD_MAX = 500.0
MEAN_MIN, MEAN_MAX = 0, 4096

USE_SPEECH = True
SPEAK_COOLDOWN = 3.0   # seconds to re-speak same class

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# -------------------------
# Speech Queue System
# -------------------------
speech_queue = queue.Queue()

def speech_worker():
    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
    engine.stop()

if USE_SPEECH:
    tts_thread = threading.Thread(target=speech_worker, daemon=True)
    tts_thread.start()

def speak(text):
    if USE_SPEECH:
        speech_queue.put(text)

# -------------------------
# Serial Helpers
# -------------------------
def list_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

def open_serial(port=None, baud=115200, timeout=1):
    if port is None:
        ports = list_ports()
        if not ports:
            raise RuntimeError("No ports found")
        port = ports[-1]

    ser = serial.Serial(port, baud, timeout=timeout)
    time.sleep(1)
    ser.reset_input_buffer()
    logging.info(f"Serial opened: {port}")
    return ser

# -------------------------
# Filters
# -------------------------
def bandpass(data, low, high, fs):
    nyq = fs*0.5
    b,a = signal.butter(4, [low/nyq, high/nyq], btype="band")
    return signal.lfilter(b,a,data)

def notch(data, freq, fs):
    w0 = freq/(fs/2)
    b,a = signal.iirnotch(w0, 30)
    return signal.lfilter(b,a,data)

def compute_spec(data):
    f, t, Sxx = signal.spectrogram(
        data, fs=SAMPLING_RATE,
        nperseg=N_PER_SEG, noverlap=N_OVERLAP
    )
    Sxx = np.log(Sxx + 1e-10)
    Sxx -= Sxx.min()
    if Sxx.max()>0:
        Sxx /= Sxx.max()
    img = cv2.resize(Sxx, IMG_SIZE)
    img = np.stack([img]*3, axis=-1)
    return img.astype(np.float32)

def valid_signal(seg):
    s = np.std(seg)
    m = np.mean(seg)
    if s<STD_MIN: return False, f"std low {s:.2f}"
    if s>STD_MAX: return False, f"std high {s:.2f}"
    if not(MEAN_MIN<=m<=MEAN_MAX): return False, f"mean bad {m:.1f}"
    return True, f"std={s:.2f}, mean={m:.1f}"

# -------------------------
# Load Model
# -------------------------
if not os.path.exists(MODEL_PATH):
    print("Model missing!")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

# -------------------------
# Serial
# -------------------------
ser = open_serial(COM_PORT, BAUDRATE)
buffer = deque(maxlen=BUFFER_SIZE)

prev_label = None
last_speak_time = 0

# -------------------------
# Plot Windows
# -------------------------
plt.ion()
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,6))

line, = ax1.plot(np.zeros(SAMPLING_RATE))
ax1.set_ylim(0,4096)
ax1.set_title("Live EEG")

img_disp = ax2.imshow(np.zeros((128,128,3)))
ax2.set_title("Spectrogram")

plt.tight_layout()

# -------------------------
# Main Loop
# -------------------------
try:
    while True:

        if plt.get_fignums()==[]:
            break

        line_in = ser.readline().decode(errors="ignore").strip()
        if not line_in:
            continue

        try:
            val = int(float(line_in))
        except:
            continue

        buffer.append(val)

        # update EEG plot
        if len(buffer)>=SAMPLING_RATE:
            last = list(buffer)[-SAMPLING_RATE:]
            line.set_ydata(last)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # wait for full window
        if len(buffer)<BUFFER_SIZE:
            continue

        seg = np.array(buffer, dtype=np.float32)

        # filtering
        seg = bandpass(seg, BANDPASS_LOW, BANDPASS_HIGH, SAMPLING_RATE)
        seg = notch(seg, NOTCH_FREQ, SAMPLING_RATE)

        ok, info = valid_signal(seg)
        if not ok:
            print("BAD SIGNAL:", info)
            buffer.clear()
            continue

        img = compute_spec(seg)
        img_disp.set_data(img)
        fig.canvas.draw()
        fig.canvas.flush_events()

        x = np.expand_dims(img, axis=0)
        preds = model.predict(x, verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx]

        label = CLASS_LABELS[idx]
        final = label if conf>=CONF_THRESHOLD else NEUTRAL_LABEL

        print(f"{final}  conf={conf:.3f}  | {info}")

        # --------------------------
        # SPEAK EVEN IF SAME LABEL
        # --------------------------
        now = time.time()
        if final != NEUTRAL_LABEL:
            if (final != prev_label) or (now - last_speak_time > SPEAK_COOLDOWN):
                speak(final)
                last_speak_time = now
                prev_label = final

        buffer.clear()

except KeyboardInterrupt:
    print("Stopped.")

finally:
    speech_queue.put(None)
    try:
        tts_thread.join()
    except:
        pass
    ser.close()
    plt.close()

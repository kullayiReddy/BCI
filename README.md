
# 🧠 Brain-Computer Interface (BCI) 

## 📌 Project Overview

This project implements a **Brain-Computer Interface (BCI) system** using the **Upside Down Labs EEG Kit** and **Arduino firmware**, integrated with real-time signal processing and machine learning to detect specific brain-signal patterns corresponding to the words:

* 😊 Happy
* 😖 Pain
* 😴 Tired

The system captures EEG signals, processes them through a web-based interface (Chords Web), extracts features, and classifies the mental state using trained ML models.

---

# 🎯 Objective

To build a real-time, low-cost, portable EEG-based system capable of:

* Capturing brain signals
* Processing and filtering EEG data
* Training ML models on specific cognitive/emotional states
* Performing real-time classification

---

# 🧰 Hardware Components

## 🧩 1. Upside Down Labs EEG Kit

![Image](https://www.crowdsupply.com/img/c05d/33ce5379-9c32-40b1-898f-c2aece48c05d/bioamp-hero_jpg_project-main.jpg)

![Image](https://docs.upsidedownlabs.tech/_images/Front_Specifications.jpg)

![Image](https://docs.upsidedownlabs.tech/_images/eeg_placement.png)

![Image](https://m.media-amazon.com/images/I/61v2v75Q73L._AC_UF1000%2C1000_QL80_.jpg)

### Includes:

* BioAmp EXG Pill (EEG amplifier)
* Electrodes
* Jumper wires
* USB cable

---

## 🔌 2. Arduino Board

* Arduino Uno (or compatible board)
* Used for:

  * Reading analog EEG signals
  * Streaming data via Serial communication

---

# 🖥 Software Stack

### 🔹 Firmware

* Arduino firmware for reading EEG signals
* Serial communication at 115200 baud rate

### 🔹 Signal Visualization

* Chords Web (Web-based EEG visualizer)
* Real-time signal plotting

### 🔹 Programming

* Python 3.x
* Libraries:

  * NumPy
  * Pandas
  * Matplotlib
  * Scikit-learn
  * SciPy

---

# ⚙️ System Architecture

```
EEG Electrodes
      ↓
BioAmp EXG Pill
      ↓
Arduino (Firmware)
      ↓
Serial Communication
      ↓
Python Signal Processing
      ↓
Feature Extraction
      ↓
Machine Learning Model
      ↓
Emotion/Word Prediction
```

---

# 🔬 Signal Processing Pipeline

1. Raw EEG acquisition
2. Noise removal

   * Band-pass filtering
   * Notch filtering (50Hz)
3. Signal normalization
4. Window segmentation
5. Feature extraction:

   * Mean
   * Standard deviation
   * FFT features
   * Power spectral density
6. Model training

---

# 🤖 Machine Learning Model

## 🏷 Training Labels

We trained the system using three cognitive/emotional words:

* Happy
* Pain
* Tired

## 📊 Training Process

1. Data collection for each state
2. Label assignment
3. Feature extraction
4. Train-test split (80-20)
5. Model training using:

   * Random Forest
   * SVM (optional comparison)

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* Confusion Matrix

---

# 🚀 How to Run the Project

## 🛠 Step 1: Hardware Setup

1. Connect BioAmp EXG Pill to Arduino.
2. Place electrodes:

   * Forehead
   * Reference electrode behind ear
3. Connect Arduino to PC via USB.

---

## 🔧 Step 2: Upload Arduino Firmware

1. Open Arduino IDE
2. Select correct COM port
3. Upload firmware code from:

```
Arduino Codes/
```

---

## 🖥 Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## ▶ Step 4: Run Data Collection Script

```bash
python collect_data.py
```

---

## 🤖 Step 5: Train Model

```bash
python train_model.py
```

---

## 🧠 Step 6: Real-Time Prediction

```bash
python predict.py
```

System will display:

```
Detected State: Happy
```

---

# 📁 Project Structure

```
BCI-Project/
│
├── Arduino Codes/
├── Python Codes/
│   ├── collect_data.py
│   ├── train_model.py
│   ├── predict.py
│
├── Models/
├── DataSets/
├── requirements.txt
└── README.md
```

---

# 📊 Results

* Achieved consistent classification between three trained states.
* Real-time detection working with minimal latency.
* Portable and low-cost implementation.

---

# 🧪 Applications

* Assistive communication systems
* Mental state monitoring
* Neurofeedback therapy
* Cognitive research
* Emotion-based human-computer interaction

---

# ⚠ Limitations

* Small dataset size
* Subject-specific training
* EEG noise sensitivity
* Requires electrode calibration

---

# 🔮 Future Improvements

* Deep Learning models (CNN for EEG)
* More emotional states
* Multi-channel EEG
* Cloud-based monitoring
* Mobile app integration

---

# 👨‍💻 Team

Final Year B.E (AI) Project
Dayananda Sagar Academy of Technology and Management
- **G Kullayireddy**  
- **Pooja Bellapukonda** 
- **S Tejo Raditya**
- **A Lakshmi poojitha**
---


# 📚 References

* Upside Down Labs Documentation
* EEG Signal Processing Research Papers
* Scikit-learn Documentation

---

# 🏁 Conclusion

This project demonstrates the practical implementation of a Brain-Computer Interface using affordable hardware and machine learning, enabling detection of specific mental states through EEG signals.



# Real-Time Sign Language Translator (ISL & ASL) 🤟🔊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe%20Hands-orange.svg)](https://developers.google.com/mediapipe)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![UI: CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-00B4D8.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision and Machine Learning system that converts dual-language sign gestures (**Indian Sign Language [ISL]** and **American Sign Language [ASL]**) into real-time text and synthesized speech.

Features real-time hand skeleton overlay tracking, prediction temporal smoothing, word/sentence builders, and an interactive glassmorphic desktop GUI.

---

## 📸 Key Features

- **Dual Sign Language Support:** Seamless runtime switching between **ISL** (Indian Sign Language, 2-handed alphabet gestures) and **ASL** (American Sign Language, single/dual hand alphabet).
- **Geometric Landmark Invariance:** Extracts 21 3D hand landmarks (84 normalized feature coordinates) via Google MediaPipe. Hand features are translation- and scale-normalized relative to bounding coordinates, ensuring invariant inference across varying camera distances.
- **High-Accuracy ML Benchmark:** Evaluated against multiple classifiers (Random Forest, SVM, KNN, Decision Tree, Logistic Regression), achieving **98.9% test accuracy** on ISL using an ensemble of 100 decision trees.
- **Temporal Stabilization Buffer:** Uses a sliding temporal queue (25 frames) to filter out spurious hand jitter and transitional frames, only registering stabilized letters.
- **Interactive Word & Sentence Builder:** Assembles letters into words, supports space/period punctuation triggers, and incorporates backspace/reset controls.
- **Audio Feedback Engine:** Non-blocking background worker thread executing `pyttsx3` text-to-speech engine to speak formed words and full sentences aloud.
- **Glassmorphic Modern UI:** Built with CustomTkinter in a dark Cyberpunk palette, displaying live skeletal overlays, confidence metrics, and camera selectors.

---

## 🧠 System Architecture

```
                       ┌────────────────────────┐
                       │   Live Camera Stream   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    MediaPipe Hands     │
                       │ (21 Landmark Detect.)  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Feature Normalization  │
                       │ (84 Normalized Coords) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Random Forest Classifier│
                       │  (ISL: 98.9% Test Acc) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Temporal Smoothing     │
                       │ (Stabilization Buffer) │
                       └───────────┬────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
  ┌───────────────────┐                         ┌───────────────────┐
  │  Word / Sentence  │                         │ Text-to-Speech    │
  │     Assembly      │                         │  (pyttsx3 Audio)  │
  └───────────────────┘                         └───────────────────┘
```

---

## 📊 Benchmark Results

Evaluated on held-out 20% stratified test splits across ISL and ASL gesture datasets:

| Model | ISL Train Acc | ISL Test Acc | ASL Train Acc | ASL Test Acc | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **100.0%** | **98.9%** | **100.0%** | **98.4%** | **Best Model (Selected)** |
| **SVM (RBF Kernel)** | 99.4% | 98.1% | 99.1% | 97.6% | Benchmark |
| **K-Nearest Neighbors (KNN)** | 98.2% | 96.5% | 97.8% | 95.8% | Benchmark |
| **Decision Tree** | 100.0% | 94.2% | 100.0% | 93.1% | Benchmark |
| **Logistic Regression** | 96.8% | 94.0% | 95.9% | 93.4% | Benchmark |

---

## 📂 Project Structure

```
Real-Time-Sign-Language-Detection/
├── CollectingData.py      # Camera capture utility for building custom sign datasets
├── CreateDataset.py       # Landmark extraction & coordinate serialization pipeline
├── TrainingModels.py      # Multi-model benchmarking and model pickle exporter
├── main.py                # Full-featured CustomTkinter desktop application
├── ISL_best_model.p       # Pre-trained Random Forest model for Indian Sign Language
├── ASL_best_model.p       # Pre-trained Random Forest model for American Sign Language
├── ISL_data.pickle        # Extracted normalized features for ISL dataset
├── ASL_data.pickle        # Extracted normalized features for ASL dataset
├── requirements.txt       # Project dependencies
└── README.md
```

---

## ⚡ Quickstart

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone the Repository
```bash
git clone https://github.com/Animesh2102/Real-Time-Sign-Language-Detection.git
cd Real-Time-Sign-Language-Detection
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

---

## 🎮 Application Controls

- **ISL / ASL Dropdown:** Switch the active machine learning classification model in real-time.
- **Camera Dropdown:** Select between integrated or external USB webcams.
- **⌫ Delete:** Remove the last registered character from the current word.
- **Reset:** Clear the current word and sentence buffers.
- **Pause / Resume:** Freeze video processing temporarily.
- **🔊 Speak:** Manually trigger audio readout of the current sentence or word.
- **Space Gesture:** Automatically speaks the completed word and commits it to the sentence.
- **Period Gesture:** Commits the sentence and concludes the statement.

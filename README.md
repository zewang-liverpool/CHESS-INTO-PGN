# ♟️ Chess-CV-Pipeline: From Raw Pixels to PGN Artifacts

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An End-to-End Machine Learning Pipeline designed to autonomously extract chess moves from standard video footage, overcoming classical computer vision challenges like Kinematic Interference and Illumination Variance.**

---

## 💡 The Vision: Solving the "Ghost Move" Dilemma
Traditional computer vision approaches in chess extraction frequently suffer from **"Ghost Move Hallucinations"**—situations where a player's hand shadow or a slight camera flicker tricks the system into recording a false move. This leads to *State Desynchronization* between the real-world board and the virtual logic engine, ultimately crashing the transcription.

**Chess-CV-Pipeline** solves this by shifting the paradigm from naive pixel-luminance differencing to **Topological Edge-Gradient Analysis**, backed by a custom-trained `ResNet-18` Deep Convolutional Neural Network.



---

## ✨ Core Architectural Innovations

* 🌟 **Illumination-Invariant Structural Validation:** Utilizes `cv2.Canny()` edge detection to analyze the physical geometry of the board. Soft shadows and lighting changes are ignored; only true topological transformations trigger the inference engine.
* 🧠 **Deep CNN Spatial Inference:** A PyTorch-powered `ResNet-18` model, fine-tuned specifically for recognizing 64-square chess topologies (Empty, Pawns, Knights, Bishops, Rooks, Queens, Kings) across diverse lighting conditions.
* 🛡️ **Strict Anti-Hallucination Protocol:** Employs a bipartite validation layer combining kinematic motion detection with a strict probabilistic confidence threshold (`Score > 1.3/2.0`). 
* 🕹️ **Interactive CLI Orchestrator:** A centralized `main.py` controller that elegantly manages automated data ingestion, human-in-the-loop annotation, and cyclic model training without touching a single line of code.
* 📜 **Standardized PGN Serialization:** Outputs fully compliant Portable Game Notation (PGN) artifacts, complete with proper metadata headers and algebraic notation, validated by the `python-chess` logic engine.

---

## 📂 Engineering Directory

```text
📦 Chess-CV-Pipeline
 ┣ 📂 chess_dataset/               # Local training data repository
 ┣ 📜 01_auto_data_collector.py    # Automated ROI slicing & data ingestion
 ┣ 📜 01_manual_data_collector.py  # Human-in-the-loop annotation tool
 ┣ 📜 02_model_trainer.py          # PyTorch ResNet-18 cyclic training module
 ┣ 📜 03_extract_static_camera.py  # 🚀 Core Inference Engine (Static Mode)
 ┣ 📜 main.py                      # Centralized CLI ML Pipeline Controller
 ┣ 📜 generate_notebook.py         # Academic presentation notebook generator
 ┣ 📜 chess_ai_model.pth           # Compiled pre-trained CNN weights
 ┗ 📜 extracted_game_static.pgn    # Output: Serialized chess artifact
🚀 Quick Start Guide
1. Environment Setup
Clone this repository and install the required MLOps dependencies:

Bash
pip install torch torchvision opencv-python python-chess numpy Pillow jupyter
2. The Command Center (Training & Ingestion)
To train the model or collect new data, launch the Pipeline UI:

Bash
python main.py
[Option 1] Automated Pipeline: Ideal for cold-starting the model with cyclic epochs.

[Option 2] Manual Pipeline: Designed for precision fine-tuning on edge cases.

3. Live Video Extraction
Run the core inference engine on your target video (chess_video.mp4):

Bash
python 03_extract_static_camera.py
⚠️ Calibration Required: Upon initialization, an interactive GUI will appear. Please meticulously click the 4 physical outer corners of the chessboard (Order: Top-Left ➔ Top-Right ➔ Bottom-Right ➔ Bottom-Left) to calculate the Homography Projection Matrix.

🔬 Under the Hood: The 3-Layer Defense System
To guarantee that the generated .pgn is flawless, the inference loop must pass three consecutive gates:

Kinematic Filter (Motion Detection): changed_pixels > 1500 ensures a physical entity (like a hand) has entered and exited the frame.

Topological Filter (Canny Edge Delta): real_change_pixels < 1500 filters out video noise and transient shadows. It updates the baseline matrix dynamically to prevent deadlock.

Probabilistic Filter (CNN Confidence): The joint probability score of the source square becoming Empty and the target square becoming the Target Piece must exceed 1.3 before committing to the logic engine.

Built with passion for Computer Vision, Deep Learning, and the timeless game of Chess. ♞
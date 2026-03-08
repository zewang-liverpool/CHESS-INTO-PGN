♟️ AI Chess Video-to-PGN Extraction System - Architecture & Technical Specification
1. Project Overview
This project is an End-to-End computer vision and deep learning pipeline. Its primary mission is to transform real-world chess match videos—which may contain human limb interference, lighting variations, and minor camera jitters—into structured, standardized digital scores in PGN (Portable Game Notation) format.

The system achieves high extraction accuracy through a four-stage cascaded algorithm: Perspective Correction -> Frame-Difference Stabilization -> Convolutional Neural Network (CNN) Recognition -> Move Legality Validation.

2. Tech Stack & Libraries
This project is built upon modern artificial intelligence and computer vision frameworks, primarily relying on the following underlying technologies:

👁️ Computer Vision
Library: cv2 (OpenCV-Python)

Applications:

Perspective Transform: Utilizing cv2.getPerspectiveTransform to map the chessboard from a 3D physical space with perspective distortion onto an absolute standard 2D 800x800 pixel grid.

Feature Tracking: In the dynamic camera module, the ORB (Oriented FAST and Rotated BRIEF) algorithm is used to extract wood-grain features of the board. A Homography matrix is calculated in real-time via BFMatcher + RANSAC to achieve robust camera tracking.

Motion Detection: Implementing Frame Differencing (cv2.absdiff) and morphological operations (cv2.dilate/cv2.erode) to accurately identify human hand occlusion, ensuring the AI performs inference only when the frame is "absolutely static."

🧠 Deep Learning
Libraries: torch, torchvision (PyTorch)

Applications:

Image Preprocessing: Utilizing the transforms module for Resizing (224x224), ColorJitter (to enhance robustness against lighting), and Tensor Normalization.

Convolutional Neural Network (CNN): Incorporating the classic ResNet-18 (Residual Network) architecture.

Transfer Learning: Loading pre-trained weights and modifying the final Fully Connected (FC) layer (nn.Linear) to transition from 1000 classes to 13 specific classes (12 piece types + 1 empty square). The model is fine-tuned using Backpropagation with the Adam optimizer.

♟️ Chess Logic Engine
Library: python-chess

Applications:

Maintaining the virtual state of the chessboard via chess.Board.

Heuristic Inference: After the AI identifies probabilities for all 64 squares, it combines these with board.legal_moves. By calculating the joint probability of "Starting square becomes empty + Destination square contains the piece," the system selects the highest-scoring legal move, preventing the AI from generating "ghost pieces" or illegal moves.

Exporting standardized PGN files using chess.pgn.FileExporter.

🛠️ Data & System Integration
Libraries: numpy (High-level matrix and vector operations), PIL (Image format bridging), os, subprocess, sys (Operating system-level process scheduling).

3. Module Descriptions
The project adopts a modular design where each script operates independently.

🎮 System Control Layer
main.py (Central Console):

Role: The primary interactive entry point.

Function: Uses the subprocess module to encapsulate the execution logic of underlying scripts, allowing users to initiate sub-functions securely without manual command-line inputs.

🧲 Phase 1: Data Preparation Layer
01_auto_data_collector.py (Automated Data Collector):

Role: A data generation pipeline based on standard opening configurations.

Function: Rectifies and slices the chessboard into 64 segments before the game starts. It automatically categorizes cropped images into 13 folders based on a predefined INITIAL_BOARD array, significantly reducing manual labeling time.

01_manual_data_collector.py (Manual Assisted Collector):

Role: A fallback solution for non-standard opening positions.

Function: Slices the board into 64 segments and saves them to a unified directory for subsequent manual classification by the user.

🧠 Phase 2: Model Training Layer
02_model_trainer.py (AI Model Training Engine):

Role: Transforms raw image data into generalized neural network weights.

Function: Loads images via PyTorch DataLoader, performs forward passes through ResNet-18, calculates Cross-Entropy Loss, and updates weights through backpropagation. After 10 epochs, the optimal parameters are serialized as chess_ai_model.pth.

🚀 Phase 3: Inference & Extraction Layer
03_extract_static_camera.py (Static Camera Extractor):

Role: A high-stability inference engine optimized for fixed-camera footage.

Function: Generates a global perspective matrix via 4-point calibration. It monitors motion using frame differencing and triggers Softmax probability prediction for all 64 squares when the frame is stable for 30 consecutive frames.

03_extract_dynamic_camera.py (Dynamic Camera Extractor):

Role: A robust engine for complex video sources involving camera movement or jitter.

Function: Features ORB-based tracking to update the transformation matrix in real-time. It employs a Dead-zone and EMA (Exponential Moving Average) filter (0.1 * New + 0.9 * Old) to suppress jitter and eliminate "Rolling Shutter" or "Jelly" effects caused by sensor noise.

📦 Utilities
generate_notebook.py (Notebook Packager):

Function: An automation script that assembles the Python modules into a .ipynb interactive document via JSON serialization for research presentation and sharing.
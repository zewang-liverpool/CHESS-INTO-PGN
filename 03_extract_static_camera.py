import cv2
import numpy as np
import chess
import chess.pgn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ================= 1. Spatial Utility Functions =================
def get_square_name(row, col):
    """Maps 2D matrix indices to standard algebraic chess notation (e.g., a1-h8)."""
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    return files[col] + ranks[row]

# ================= 2. Hyperparameters & Deep Learning Initialization =================
VIDEO_PATH = 'chess_video.mp4'
START_TIME_SEC = 54  # Offset to bypass non-gameplay initialization footage
MODEL_PATH = 'chess_ai_model.pth'
CLASSES = ['Empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

# Hardware acceleration mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate ResNet-18 architecture and adapt the fully connected (fc) layer
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval() # Set to evaluation mode (disable dropout/batchnorm updates)
    print("[System] Pre-trained CNN weights loaded successfully.")
else:
    print("[Error] Model weights not found. Execute the training pipeline first.")
    exit()

# Image preprocessing tensor pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 3. Spatial Calibration (Homography Matrix) =================
clicked_points = []
def mouse_callback(event, x, y, flags, param):
    """Event listener for interactive 4-point homography calibration."""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", param)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, START_TIME_SEC * fps)

ret, first_frame = cap.read()
if not ret:
    print("[Error] Video stream initialization failed.")
    exit()
    
calib_frame = first_frame.copy()
cv2.imshow("Calibration", calib_frame)
cv2.setMouseCallback("Calibration", mouse_callback, calib_frame)
print("\n[Action Required] Define the Region of Interest (ROI).")
print("Select the 4 corners of the board: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left.")
cv2.waitKey(0)
cv2.destroyWindow("Calibration")

if len(clicked_points) != 4:
    print("[System] Calibration aborted by user.")
    exit()

# Compute the static perspective transformation matrix (M_orig)
pts_src = np.array(clicked_points, dtype=np.float32)
pts_dst = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
M_orig = cv2.getPerspectiveTransform(pts_src, pts_dst)

# ================= 4. Logic Engine & PGN Metadata Initialization =================
board = chess.Board()
game = chess.pgn.Game()

# Injecting standard metadata to prevent "[Event "?"]" headers in the output PGN
game.headers["Event"] = "Computer Vision Chess Extraction Test"
game.headers["Site"] = "Laboratory Environment"
game.headers["Date"] = "2026.03.11"
game.headers["White"] = "Player A"
game.headers["Black"] = "Player B"

node = game 

prev_gray = None
last_board_gray = None # Tracks the baseline structural state of the board
stable_frames = 0
is_moving = False

print("\n========================================")
print("Pipeline Active: Static Extractor with Edge-Gradient Filtering")
print("Monitoring temporal pixel variations...")
print("========================================")

# ================= 5. Main Extractor Loop =================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Apply homography projection to rectify perspective distortion
    curr_warped = cv2.warpPerspective(frame, M_orig, (800, 800))
    curr_gray = cv2.GaussianBlur(cv2.cvtColor(curr_warped, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    
    # Initialize baseline tensors on frame 0
    if prev_gray is None:
        prev_gray = curr_gray
        last_board_gray = curr_gray.copy() 
        continue

    # --- Phase 1: Kinematic Detection (Temporal Differencing) ---
    # Detects gross movement (e.g., human hand entering the frame)
    diff = cv2.absdiff(prev_gray, curr_gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(cv2.erode(thresh, kernel, iterations=2), kernel, iterations=2)
    
    changed_pixels = cv2.countNonZero(dilated)
    
    # State Machine: High variation implies active human interference
    if changed_pixels > 1500:
        is_moving = True
        stable_frames = 0
        cv2.putText(curr_warped, "KINEMATIC INTERFERENCE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        stable_frames += 1
        cv2.putText(curr_warped, f"STABILIZATION: {stable_frames}/30", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Trigger validation upon achieving temporal stability
        if stable_frames == 30 and is_moving:
            is_moving = False
            
            # --- Phase 2: Illumination-Invariant Structural Validation ---
            # Extract topological edges using Canny operator to ignore soft shadows
            edges_last = cv2.Canny(last_board_gray, 50, 150)
            edges_curr = cv2.Canny(curr_gray, 50, 150)
            
            # Compute topological gradient delta
            board_diff = cv2.absdiff(edges_last, edges_curr)
            
            # Dilate edges slightly to account for minor camera micro-vibrations
            kernel_edge = np.ones((3, 3), np.uint8)
            board_thresh = cv2.dilate(board_diff, kernel_edge, iterations=1)
            
            real_change_pixels = cv2.countNonZero(board_thresh)
            
            # Filter condition: Threshold set to 1500 (edges contain far fewer pixels than areas)
            if real_change_pixels < 1500:
                print(f"[Filtered] Illumination/Shadow variance rejected. Edge delta: {real_change_pixels} px")
                last_board_gray = curr_gray.copy() # Update baseline to prevent progressive drift
                continue 
                
            print(f"\n[Trigger] Topological structure modified (Edge delta: {real_change_pixels} px). Executing CNN inference...")
            
            # --- Phase 3: CNN Spatial Inference Pipeline ---
            board_probs = {}
            for r in range(8):
                for c in range(8):
                    # Spatial cropping with vertical expansion for piece height
                    y1, y2 = max(0, r*100 - 80), (r+1)*100
                    x1, x2 = max(0, c*100 + 5), min(800, (c+1)*100 - 5)
                    sq_img = curr_warped[y1:y2, x1:x2]
                    
                    img_rgb = cv2.cvtColor(sq_img, cv2.COLOR_BGR2RGB)
                    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        out = model(tensor)
                        probs = torch.nn.functional.softmax(out[0], dim=0)
                        board_probs[get_square_name(r, c)] = {CLASSES[i]: probs[i].item() for i in range(len(CLASSES))}
            
            # --- Phase 4: Probabilistic Move Inference & Logical Validation ---
            best_move, max_score = None, -1
            for move in board.legal_moves:
                s_sq = chess.square_name(move.from_square)
                e_sq = chess.square_name(move.to_square)
                p_sym = board.piece_at(move.from_square).symbol()
                
                # Joint probability score: Source emptied + Target occupied (Max: 2.0)
                score = board_probs[s_sq]['Empty'] + board_probs[e_sq][p_sym]
                
                if score > max_score:
                    max_score, best_move = score, move
            
            # Confidence threshold set to 1.3 based on empirical tuning
            if max_score > 1.3:
                board.push(best_move)
                node = node.add_variation(best_move)
                print(f"[Success] Move registered: {best_move.uci()} | Prob Score: {max_score:.2f}")
                
                last_board_gray = curr_gray.copy() # Commit structural baseline
                
                # Real-time PGN checkpointing to prevent data loss on unexpected termination
                with open("extracted_game_static.pgn", "w", encoding="utf-8") as f:
                    exporter = chess.pgn.FileExporter(f)
                    game.accept(exporter)
            else:
                print(f"[Rejected] Hallucination suppressed. Prob Score: {max_score:.2f} (Below 1.3 threshold).")
                last_board_gray = curr_gray.copy() # Force baseline update to break deadlocks

    cv2.imshow('CNN Extraction Stream', curr_warped)
    prev_gray = curr_gray
    
    if cv2.waitKey(1) & 0xFF == 27: break

# ================= 6. Resource Deallocation =================
cap.release()
cv2.destroyAllWindows()

print("\n[System] Pipeline terminated gracefully. Exporting final serialization...")
with open("extracted_game_static.pgn", "w", encoding="utf-8") as f:
    exporter = chess.pgn.FileExporter(f)
    game.accept(exporter)
print("[System] Artifact saved successfully to: extracted_game_static.pgn")
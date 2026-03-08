import cv2
import numpy as np
import chess
import chess.pgn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ================= Helper Functions =================
def get_square_name(row, col):
    """Convert grid row and column indices to standard algebraic notation (a1-h8)."""
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    return files[col] + ranks[row]

# ================= 1. Configuration and Model Initialization =================
VIDEO_PATH = 'chess_video.mp4'
START_TIME_SEC = 54  # Skip the initial video segment without active gameplay
MODEL_PATH = 'chess_ai_model.pth'
CLASSES = ['Empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print("Success: Model weights loaded successfully.")
else:
    print("Error: Model file not found. Please execute the training script first.")
    exit()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 2. Initial Static Calibration =================
clicked_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", param)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, START_TIME_SEC * fps)

ret, first_frame = cap.read()
if not ret:
    print("Error: Failed to read the video stream.")
    exit()
    
calib_frame = first_frame.copy()
cv2.imshow("Calibration", calib_frame)
cv2.setMouseCallback("Calibration", mouse_callback, calib_frame)
print("\nPlease select the four physical corners of the chessboard (Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left).")
cv2.waitKey(0)
cv2.destroyWindow("Calibration")

if len(clicked_points) != 4:
    print("Calibration cancelled.")
    exit()

# Compute the static perspective transformation matrix
pts_src = np.array(clicked_points, dtype=np.float32)
pts_dst = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
M_orig = cv2.getPerspectiveTransform(pts_src, pts_dst)

# ================= 3. Inference Engine Initialization =================
board = chess.Board()
game = chess.pgn.Game()
node = game 

prev_gray = None
stable_frames = 0
is_moving = False

print("\n========================================")
print("Static camera extraction mode initialized.")
print("Interference variables eliminated. Awaiting frame stabilization for inference...")
print("========================================")

# ================= 4. Main Execution Loop =================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Apply the static perspective transform to rectify the chessboard image
    curr_warped = cv2.warpPerspective(frame, M_orig, (800, 800))
    curr_gray = cv2.GaussianBlur(cv2.cvtColor(curr_warped, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    
    if prev_gray is None:
        prev_gray = curr_gray
        continue

    # --- Motion Detection (Restricted to the 800x800 rectified board area) ---
    diff = cv2.absdiff(prev_gray, curr_gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(cv2.erode(thresh, kernel, iterations=2), kernel, iterations=2)
    
    changed_pixels = cv2.countNonZero(dilated)
    
    # Threshold: Pixel change count > 1500 indicates motion (e.g., hand occlusion)
    if changed_pixels > 1500:
        is_moving = True
        stable_frames = 0
        cv2.putText(curr_warped, "MOTION DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        stable_frames += 1
        cv2.putText(curr_warped, f"STABLE: {stable_frames}/30", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Trigger inference when the frame remains stable for 30 consecutive frames (approx. 1 second) post-motion
        if stable_frames == 30 and is_moving:
            is_moving = False
            print("\nFrame stabilized. Initiating model inference...")
            
            board_probs = {}
            for r in range(8):
                for c in range(8):
                    # Bounding box cropping logic: expand upper boundary to capture the piece tops
                    y1, y2 = max(0, r*100 - 80), (r+1)*100
                    x1, x2 = max(0, c*100 + 5), min(800, (c+1)*100 - 5)
                    sq_img = curr_warped[y1:y2, x1:x2]
                    
                    img_rgb = cv2.cvtColor(sq_img, cv2.COLOR_BGR2RGB)
                    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(tensor)
                        probs = torch.nn.functional.softmax(out[0], dim=0)
                        board_probs[get_square_name(r, c)] = {CLASSES[i]: probs[i].item() for i in range(len(CLASSES))}
            
            best_move, max_score = None, -1
            for move in board.legal_moves:
                s_sq, e_sq = move.uci()[:2], move.uci()[2:4]
                p_sym = board.piece_at(move.from_square).symbol()
                score = board_probs[s_sq]['Empty'] + board_probs[e_sq][p_sym]
                
                if score > max_score:
                    max_score, best_move = score, move
            
            if max_score > 1.2:
                board.push(best_move)
                node = node.add_variation(best_move)
                print(f"Move inferred: {best_move.uci()} (Confidence score: {max_score:.2f})")

    cv2.imshow('Static Inference View', curr_warped)
    prev_gray = curr_gray
    
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

print("\nExtraction complete. Generating PGN file...")
with open("extracted_game_static.pgn", "w", encoding="utf-8") as f:
    exporter = chess.pgn.FileExporter(f)
    game.accept(exporter)
print("Success: Game successfully exported to extracted_game_static.pgn.")
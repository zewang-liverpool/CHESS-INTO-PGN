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
START_TIME_SEC = 54  # Skip the initial segment without active gameplay
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
    print("Error: Model file not found. Please train the model first.")
    exit()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 2. Interactive Initial Calibration =================
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
    print("Error: Unable to read video stream.")
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

pts_src = np.array(clicked_points, dtype=np.float32)
pts_dst = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
M_orig = cv2.getPerspectiveTransform(pts_src, pts_dst)

# ================= 3. Initialize Feature Tracking and Stabilization =================
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32(clicked_points), 255)

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(first_gray, mask=mask)

current_matrix = M_orig.copy() # The current smoothed homography matrix
board = chess.Board()
game = chess.pgn.Game()
node = game 

prev_gray = None
stable_frames = 0
is_moving = False

print("\n========================================")
print("Feature tracking and temporal stabilization initialized.")
print("========================================")

# ================= 4. Main Execution Loop =================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- Dynamic Tracking and Stabilization Logic ---
    kp_cur, des_cur = orb.detectAndCompute(curr_frame_gray, None)
    if des_cur is not None and len(kp_cur) > 10:
        matches = bf.match(des_ref, des_cur)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) >= 10:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                try:
                    new_matrix_raw = M_orig @ np.linalg.inv(H)
                    
                    # Compute displacement to determine if a matrix update is necessary
                    # Calculate the L2 norm of the corner displacements
                    corners = np.float32([[0,0], [799,0], [799,799], [0,799]]).reshape(-1, 1, 2)
                    old_corners = cv2.perspectiveTransform(corners, current_matrix)
                    new_corners = cv2.perspectiveTransform(corners, new_matrix_raw)
                    move_dist = np.max(np.linalg.norm(new_corners - old_corners, axis=2))
                    
                    # Update the matrix only if the maximum displacement exceeds the threshold (e.g., 2.0 pixels)
                    if move_dist > 2.0:
                        # Apply Exponential Moving Average (EMA) to smooth the homography matrix and mitigate micro-jitter
                        current_matrix = 0.1 * new_matrix_raw + 0.9 * current_matrix
                        
                except np.linalg.LinAlgError:
                    pass

    # ---------------------------------------------
    curr_warped = cv2.warpPerspective(frame, current_matrix, (800, 800))
    curr_gray = cv2.GaussianBlur(cv2.cvtColor(curr_warped, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    
    if prev_gray is None:
        prev_gray = curr_gray
        continue

    # --- Motion Detection ---
    diff = cv2.absdiff(prev_gray, curr_gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(cv2.erode(thresh, kernel, iterations=2), kernel, iterations=2)
    
    # Calculate the number of changed pixels for motion thresholding
    changed_pixels = cv2.countNonZero(dilated)
    
    if changed_pixels > 1500:
        is_moving = True
        stable_frames = 0
        cv2.putText(curr_warped, "MOTION DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        stable_frames += 1
        cv2.putText(curr_warped, f"STABLE: {stable_frames}/30", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        if stable_frames == 30 and is_moving:
            is_moving = False
            print("\nFrame stabilized. Initiating model inference...")
            
            board_probs = {}
            for r in range(8):
                for c in range(8):
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

    cv2.imshow('Stabilized AI View', curr_warped)
    prev_gray = curr_gray
    
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

print("\nExtraction complete. Generating PGN file...")
with open("extracted_game_v6.pgn", "w", encoding="utf-8") as f:
    # Note: If python-chess version >= 1.7.0, use game.accept(chess.pgn.FileExporter(f)) instead.
    chess.pgn.FileExporter(f).visit(game)
print("Success: Game successfully exported to extracted_game_v6.pgn.")
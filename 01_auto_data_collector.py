import cv2
import numpy as np
import os

# ================= 1. Initialize output directories for 13 classes =================
save_dir = "chess_dataset"
classes = [
    "00_Empty", "01_P", "02_N", "03_B", "04_R", "05_Q", "06_K",
    "07_p_black", "08_n_black", "09_b_black", "10_r_black", "11_q_black", "12_k_black"
]
for cls in classes:
    os.makedirs(os.path.join(save_dir, cls), exist_ok=True)

# ================= 2. Standard initial board configuration =================
INITIAL_BOARD = [
    ['10_r_black', '08_n_black', '09_b_black', '11_q_black', '12_k_black', '09_b_black', '08_n_black', '10_r_black'],
    ['07_p_black'] * 8,
    ['00_Empty'] * 8,
    ['00_Empty'] * 8,
    ['00_Empty'] * 8,
    ['00_Empty'] * 8,
    ['01_P'] * 8,
    ['04_R', '02_N', '03_B', '05_Q', '06_K', '03_B', '02_N', '04_R']
]

# ================= 3. Calibration phase =================
clicked_points = []
calibration_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        cv2.circle(calibration_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", calibration_frame)

cap = cv2.VideoCapture('chess_video.mp4')
print("\nPress SPACE to pause/resume. Press 'c' to capture a clear chessboard frame for calibration...")
paused = False
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret: break
    cv2.imshow("Video Player", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord(' '): paused = not paused
    elif key == ord('c'):
        calibration_frame = frame.copy()
        break
cv2.destroyWindow("Video Player")

cv2.imshow("Calibration", calibration_frame)
cv2.setMouseCallback("Calibration", mouse_callback)
print("Please click the four actual corners of the chessboard (Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left)")
cv2.waitKey(0)
cv2.destroyAllWindows()

width, height = 800, 800
pts_src = np.array(clicked_points, dtype=np.float32)
pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# ================= 4. Automated data collection and cropping =================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("\nReplaying video...")
print("Instruction: Press the 's' key 3 to 5 times between 00:00 and 01:03 when the frame is completely static without hand occlusions.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
        
    curr_warped = cv2.warpPerspective(frame, matrix, (width, height))
    cv2.imshow('Auto Labeler', curr_warped)
    
    key = cv2.waitKey(30) & 0xFF
    if key == 27: break
    elif key == ord('s'):
        frame_count += 1
        sq_size = 100
        saved = 0
        for r in range(8):
            for c in range(8):
                # Extended bounding box logic: expand the upper boundary to preserve the tops of the chess pieces
                y1 = max(0, r * sq_size - 80) 
                y2 = (r + 1) * sq_size
                x1 = max(0, c * sq_size + 5)
                x2 = min(width, (c + 1) * sq_size - 5)
                
                square_img = curr_warped[y1:y2, x1:x2]
                folder_name = INITIAL_BOARD[r][c]
                filename = os.path.join(save_dir, folder_name, f"auto_{frame_count}_r{r}_c{c}.jpg")
                
                cv2.imwrite(filename, square_img)
                saved += 1
        print(f"Extraction {frame_count} completed: 64 squares successfully cropped and classified.")

cap.release()
cv2.destroyAllWindows()
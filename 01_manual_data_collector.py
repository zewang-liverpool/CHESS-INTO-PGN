import cv2
import numpy as np
import os

# Create a directory to store the dataset
save_dir = "chess_dataset"
os.makedirs(save_dir, exist_ok=True)

# ================= Helper functions and calibration =================
clicked_points = []
calibration_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        cv2.circle(calibration_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", calibration_frame)

cap = cv2.VideoCapture('chess_video.mp4')
if not cap.isOpened():
    print("Error: Unable to open the video file. Please ensure the file exists in the specified path.")
    exit()

print("\nPress SPACE to pause/resume playback. Press 'c' to capture a clear frame of the chessboard...")
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
    elif key == 27: exit()
cv2.destroyWindow("Video Player")

cv2.imshow("Calibration", calibration_frame)
cv2.setMouseCallback("Calibration", mouse_callback)
print("Please click the four actual corners of the chessboard (Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left).")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:
    print("Calibration cancelled due to insufficient coordinate points.")
    exit()

width, height = 800, 800
pts_src = np.array(clicked_points, dtype=np.float32)
pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# ================= Start data collection =================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("\nReplaying video sequence...")
print("Instruction: Press the 's' key to save data when the frame is clear and free of hand occlusions.")
print("Capture 5-10 different board states. Press 'Esc' to terminate the process.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
        
    curr_warped = cv2.warpPerspective(frame, matrix, (width, height))
    cv2.imshow('Data Collector', curr_warped)
    
    key = cv2.waitKey(30) & 0xFF
    if key == 27: # Exit on 'Esc'
        break
    elif key == ord('s'):
        # Extract the 64 squares upon triggering the capture command
        frame_count += 1
        sq_size = 100
        saved = 0
        for r in range(8):
            for c in range(8):
                # Apply a slight margin crop to focus the extraction on the chess piece geometry
                y1, y2 = r * sq_size + 10, (r + 1) * sq_size
                x1, x2 = c * sq_size + 10, (c + 1) * sq_size - 10
                square_img = curr_warped[y1:y2, x1:x2]
                
                filename = os.path.join(save_dir, f"frame{frame_count}_r{r}_c{c}.jpg")
                cv2.imwrite(filename, square_img)
                saved += 1
        print(f"Extraction {frame_count} completed: {saved} square images successfully written to disk.")

cap.release()
cv2.destroyAllWindows()
print(f"\nData collection sequence terminated. Please verify the outputs in the '{save_dir}' directory.")
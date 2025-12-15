import cv2
from ultralytics import YOLO
import pandas as pd
import torch
import numpy as np
import time

# -----------------------------
# CONFIGURATION ⚙️
# --- Paths and System Setup ---
# -----------------------------
VIDEO_PATH =  r"C:\Users\ajayb\Downloads\INPUT-3.mp4"#r"C:\Users\ajayb\Downloads\INPUT_VIDEO.mp4" # Path to the input CCTV video file
EXCEL_PATH = r"C:\Users\ajayb\Downloads\vehicle_counts.xlsx" # Path to save the final vehicle counts

# --- Display Settings ---``
DISPLAY_WIDTH = 1280 # Width for the output display window
DISPLAY_HEIGHT = 720 # Height for the output display window

# --- Model and Detection Thresholds ---
CONF_THRESHOLD = 0.55 # Minimum confidence score to consider a detection valid
IOU_THRESHOLD = 0.45 # Intersection over Union threshold for Non-Max Suppression

# --- Counting Line Setup ---
LINE_POSITION = 1500 # The Y-coordinate (vertical position) of the counting line

# --- Vehicle Class Mapping (COCO Dataset IDs) ---
VEHICLE_CLASSES = {
    2: "car", 
    3: "motorbike", 
    5: "bus", 
    7: "truck"
}

# --- Filtering and Size Constraints ---
MIN_WIDTH = 50 # Minimum bounding box width to filter out noise/false positives
MIN_HEIGHT = 50 # Minimum bounding box height to filter out noise/false positives

# -----------------------------
# BIDIRECTIONAL COUNTING LOGIC CONFIGURATION 🔢
# -----------------------------
# Dictionary to track the counting status of each unique vehicle ID
# Key: Track ID (int), Value: {'y_history': list of y-centers, 'counted': bool, 'direction': 'incoming' or 'outgoing' or None}
vehicle_status = {}

# Dictionaries to store the final counts aggregated by vehicle type, separated by direction
# Assuming: Incoming = Downward (y increases), Outgoing = Upward (y decreases)
vehicle_count = {
    'incoming': {name: 0 for name in VEHICLE_CLASSES.values()},
    'outgoing': {name: 0 for name in VEHICLE_CLASSES.values()}
}

# -----------------------------
# LOAD MODEL & INITIALIZE 🧠
# -----------------------------
# Determine the best device available (GPU/CUDA preferred for speed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load the YOLOv8 Large model ('l')
model = YOLO("yolov8l.pt") 
model.to(DEVICE)

# Initialize video capture object
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

# ----------------------------------------------------
# 💡 TEXT SIZE MODIFICATION 💡
# Increased the base scale significantly for a larger scoreboard.
# TARGET_FONT_SCALE is now 1.2 (previously around 0.6-0.8 equivalent).
# ----------------------------------------------------
BASE_FONT_SCALE = 1.2 
TEXT_THICKNESS = 3 # Increased thickness for better visibility
# ----------------------------------------------------

# -----------------------------
# PROCESS VIDEO FRAME-BY-FRAME 🎥
# -----------------------------
print("Starting video processing...")
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break
    
    frame_counter += 1
    
    # 1. Run YOLOv8 Tracking
    results = model.track(
        frame, 
        device=DEVICE, 
        conf=CONF_THRESHOLD, 
        iou=IOU_THRESHOLD,
        tracker="bytetrack.yaml", # Robust tracker for stable ID assignment
        persist=True,
        verbose=False 
    )

    # Draw the counting line (Y-coordinate 500)
    # Color: Yellow (0, 255, 255) | Thickness: 3
    cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 255, 255), 3)

    
    # 2. Extract and Process Detections
    track_ids = []
    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        for box, track_id, cls, conf in zip(boxes, track_ids, class_ids, confs):
            
            # Extract box coordinates, size, and center point
            x1, y1, x2, y2 = map(int, box)
            width, height = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Apply size filtering and check if class is a vehicle
            if cls not in VEHICLE_CLASSES or width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            vehicle_name = VEHICLE_CLASSES[cls]

            # 3. Draw Bounding Box & Label
            color = (0, 255, 0) if track_id in vehicle_status else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label scale remains smaller for the bounding box
            label = f"ID.{track_id} {vehicle_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) 
            
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 4. Bidirectional Counting Logic
            
            if track_id not in vehicle_status:
                vehicle_status[track_id] = {'y_history': [cy], 'counted': False, 'direction': None}
                
            status = vehicle_status[track_id]
            y_history = status['y_history']
            is_counted = status['counted']

            y_history.append(cy)
            if len(y_history) > 5: 
                y_history.pop(0)

            previous_cy = y_history[-2] if len(y_history) > 1 else cy

            if not is_counted:
                
                # --- INCOMING (DOWNWARD) Logic ---
                if previous_cy < LINE_POSITION and cy >= LINE_POSITION:
                    status['direction'] = 'incoming'
                    vehicle_count['incoming'][vehicle_name] += 1
                    status['counted'] = True

                # --- OUTGOING (UPWARD) Logic ---
                elif previous_cy > LINE_POSITION and cy <= LINE_POSITION:
                    status['direction'] = 'outgoing'
                    vehicle_count['outgoing'][vehicle_name] += 1
                    status['counted'] = True

            y_history[-1] = cy


    # 5. Display Counts and Information (LARGER FONT)
    
    # --- Incoming Count Board (Top Left) ---
    y_offset = 30 # Starting Y offset for text
    box_x_start, box_x_end = 10, 400 # Widened box for large text
    
    # Draw a background box for INCOMING counts
    cv2.rectangle(frame, (box_x_start - 5, y_offset - 25), (box_x_end, y_offset + 30 + len(VEHICLE_CLASSES) * 45), (20, 20, 20), -1)
    
    cv2.putText(frame, "INCOMING ", (box_x_start, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, BASE_FONT_SCALE * 1.2, (0, 255, 255), TEXT_THICKNESS) # Title
    y_offset += 45 # Increased spacing
    
    incoming_total = sum(vehicle_count['incoming'].values())
    cv2.putText(frame, f"TOTAL: {incoming_total}", (box_x_start, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, BASE_FONT_SCALE, (0, 255, 0), TEXT_THICKNESS) # Total
    y_offset += 35
    
    for name, count in vehicle_count['incoming'].items():
        cv2.putText(frame, f"{name}: {count}", (box_x_start, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, BASE_FONT_SCALE * 0.9, (255, 255, 255), TEXT_THICKNESS - 1) # Individual
        y_offset += 30
        
    # --- Outgoing Count Board (Top Right) ---
    x_offset_outgoing = frame.shape[1] - 400 # Adjusted start position for wider text
    y_offset = 30 # Reset Y offset for top of screen
    
    # Draw a background box for OUTGOING counts
    cv2.rectangle(frame, (x_offset_outgoing - 5, y_offset - 25), (frame.shape[1] - 10, y_offset + 30 + len(VEHICLE_CLASSES) * 45), (20, 20, 20), -1)
    
    cv2.putText(frame, "OUTGOING ", (x_offset_outgoing, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, BASE_FONT_SCALE * 1.2, (0, 255, 255), TEXT_THICKNESS) # Title
    y_offset += 45
    
    outgoing_total = sum(vehicle_count['outgoing'].values())
    cv2.putText(frame, f"TOTAL: {outgoing_total}", (x_offset_outgoing, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, BASE_FONT_SCALE, (0, 255, 0), TEXT_THICKNESS) # Total
    y_offset += 35
    
    for name, count in vehicle_count['outgoing'].items():
        cv2.putText(frame, f"{name}: {count}", (x_offset_outgoing, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, BASE_FONT_SCALE * 0.9, (255, 255, 255), TEXT_THICKNESS - 1) # Individual
        y_offset += 30


    # Resize frame for display
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Bidirectional Vehicle Counting & Tracking (Large Text)", frame_resized)
    
    # 6. Garbage Collection (Cleanup of Stale Tracks)
    current_ids = set(track_ids)
    keys_to_delete = [id for id in vehicle_status if id not in current_ids and vehicle_status[id]['counted']]
    for id in keys_to_delete:
        del vehicle_status[id]


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP & SAVE RESULTS 💾
# -----------------------------
cap.release()
cv2.destroyAllWindows()

# Merge and save final results to the specified Excel file
final_data = {
    'Direction': [],
    'Vehicle Type': [],
    'Count': []
}

for direction, counts in vehicle_count.items():
    for name, count in counts.items():
        final_data['Direction'].append(direction.capitalize())
        final_data['Vehicle Type'].append(name.capitalize())
        final_data['Count'].append(count)
    # Add a row for the total for clarity
    final_data['Direction'].append(direction.capitalize())
    final_data['Vehicle Type'].append('TOTAL')
    final_data['Count'].append(sum(counts.values()))

df = pd.DataFrame(final_data)
try:
    df.to_excel(EXCEL_PATH, index=False)
    print("\n--- Final Vehicle Counts ---")
    print(df.to_string(index=False))
    print(f"\nResults successfully saved to {EXCEL_PATH}")
except Exception as e:
    print(f"Error saving to Excel: {e}")
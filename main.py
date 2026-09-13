import os
import time
import urllib.request
from collections import deque
import cv2
import numpy as np
import pyvirtualcam

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download Model File if missing
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# 3D Unit Cube Setup
size = 0.8
cube_vertices = np.array([
    [-size, -size, -size], [ size, -size, -size], [ size,  size, -size], [-size,  size, -size],
    [-size, -size,  size], [ size, -size,  size], [ size,  size,  size], [-size,  size,  size]
], dtype=np.float32)

faces = [
    ([0, 1, 2, 3], (180, 50, 50)),   # Back face
    ([4, 5, 6, 7], (50, 180, 50)),   # Front face
    ([0, 3, 7, 4], (50, 50, 180)),   # Left face
    ([1, 2, 6, 5], (200, 200, 50)),  # Right face
    ([0, 1, 5, 4], (200, 50, 200)),  # Bottom face
    ([3, 2, 6, 7], (50, 200, 200))   # Top face
]

# Rotation & Velocity State
curr_rot_x, curr_rot_y = 0.0, 0.0
ang_vel_x, ang_vel_y = 0.0, 0.0  # Radians per second

target_pos_x, target_pos_y, pos_z = 0.0, 0.0, 4.0
curr_pos_x, curr_pos_y = 0.0, 0.0

is_pinching = False
was_pinching = False
two_hand_start_time = None

# Flick Buffer to calculate launch impulse velocity (stores last 5 frames: (pos, time))
pinch_history = deque(maxlen=5)

# Tuning Parameters
PINCH_ENTER_THRESH = 0.065   # Distance to start pinch
PINCH_EXIT_THRESH  = 0.095   # Hysteresis threshold to maintain pinch
FLICK_IMPULSE_GAIN = 0.0035  # Magnifies hand velocity into spin impulse on release
DRAG_FRICTION      = 0.85    # Resistance while actively pinching (tight control)
FREE_SPIN_FRICTION = 0.985   # Resistance after flick release (basketball spin effect)
REQUIRED_HOLD_TIME = 0.3

def get_rotation_matrix(rx, ry):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    return Ry @ Rx

cap = cv2.VideoCapture(0)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

prev_frame_time = time.time()

with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        dt = max(current_time - prev_frame_time, 0.001)
        prev_frame_time = current_time

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        focal_length = w

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect_for_video(mp_image, int(current_time * 1000))

        hand_centers = []
        active_pinch = None

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                thumb, index, wrist = hand_landmarks[4], hand_landmarks[8], hand_landmarks[0]
                pinch_dist = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2 + (thumb.z - index.z)**2)
                
                hand_centers.append(np.array([wrist.x, wrist.y, wrist.z]))

                threshold = PINCH_EXIT_THRESH if is_pinching else PINCH_ENTER_THRESH
                if pinch_dist < threshold:
                    is_pinching = True
                    active_pinch = np.array([index.x * w, index.y * h])
                else:
                    is_pinching = False

        num_hands = len(hand_centers)

        # 1. Two-Hand Movement Logic
        if num_hands == 2:
            pinch_history.clear()
            if two_hand_start_time is None:
                two_hand_start_time = current_time
            elif current_time - two_hand_start_time >= REQUIRED_HOLD_TIME:
                midpoint = (hand_centers[0] + hand_centers[1]) / 2.0
                target_pos_x = (midpoint[0] - 0.5) * 4.0
                target_pos_y = (midpoint[1] - 0.5) * 3.0
        else:
            two_hand_start_time = None

            # 2. Single-Hand Pinch & Flick Tracking
            if num_hands == 1 and active_pinch is not None:
                # Store position & timestamp history during pinch
                pinch_history.append((active_pinch, current_time))
                
                # Active drag rotation tracking
                if len(pinch_history) >= 2:
                    p_curr, t_curr = pinch_history[-1]
                    p_prev, t_prev = pinch_history[-2]
                    time_delta = max(t_curr - t_prev, 0.001)
                    
                    # Direct responsive drag velocity
                    ang_vel_y = ((p_curr[0] - p_prev[0]) / time_delta) * 0.002
                    ang_vel_x = ((p_curr[1] - p_prev[1]) / time_delta) * 0.002
            else:
                # 3. Detect FLICK RELEASE Event (Hand just unpinched or left tracking)
                if was_pinching and len(pinch_history) >= 2:
                    # Calculate release impulse from full trajectory buffer
                    p_start, t_start = pinch_history[0]
                    p_end, t_end = pinch_history[-1]
                    total_dt = max(t_end - t_start, 0.001)

                    # Compute flick launch velocity vector
                    flick_vel_x = (p_end[0] - p_start[0]) / total_dt
                    flick_vel_y = (p_end[1] - p_start[1]) / total_dt

                    # Apply impulse to angular velocity
                    ang_vel_y = flick_vel_x * FLICK_IMPULSE_GAIN
                    ang_vel_x = flick_vel_y * FLICK_IMPULSE_GAIN

                pinch_history.clear()

        was_pinching = is_pinching and (num_hands == 1)

        # 4. Physics & Dual Friction Engine
        curr_rot_x += ang_vel_x * dt
        curr_rot_y += ang_vel_y * dt
        
        # Apply lower friction when free-spinning to simulate basketball inertia
        current_friction = DRAG_FRICTION if is_pinching else FREE_SPIN_FRICTION
        ang_vel_x *= current_friction
        ang_vel_y *= current_friction

        curr_pos_x += (target_pos_x - curr_pos_x) * 0.2
        curr_pos_y += (target_pos_y - curr_pos_y) * 0.2

        # 5. Projection & Rendering
        R = get_rotation_matrix(curr_rot_x, curr_rot_y)
        T = np.array([curr_pos_x, curr_pos_y, pos_z], dtype=np.float32)
        cam_vertices = cube_vertices @ R.T + T

        projected_2d = []
        for point in cam_vertices:
            z = max(point[2], 0.1)
            projected_2d.append([int(focal_length * (point[0] / z) + cx), int(focal_length * (point[1] / z) + cy)])
        projected_2d = np.array(projected_2d, dtype=np.int32)

        sorted_faces = []
        for indices, color in faces:
            avg_z = np.mean(cam_vertices[indices, 2])
            sorted_faces.append((avg_z, indices, color))
        sorted_faces.sort(key=lambda item: item[0], reverse=True)

        overlay = frame.copy()
        for _, indices, color in sorted_faces:
            pts = projected_2d[indices]
            cv2.fillConvexPoly(overlay, pts, color)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cam.send(frame)
        cam.sleep_until_next_frame()

        cv2.imshow("AR Engine Debugger", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
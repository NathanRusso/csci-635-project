# Logging Imports
print("[INFO] Importing libraries...")
import logging
import os
import warnings

# Suppress warnings on run
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppress TensorFlow startup messages
warnings.filterwarnings("ignore")           # Suppress Python/Keras warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)     # Suppress TensorFlow logging
original_stderr = os.dup(2)                 # Temporarily suppress C++ startup output
devnull = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(devnull, 2)
    # import tensorflow as tf     # Note: Takes a few seconds
    import keras
finally:
    os.dup2(original_stderr, 2)
    os.close(original_stderr)
    os.close(devnull)

import cv2 as cv
import numpy as np

import config as cf
import classifier2

ASPECT_RATIO = 2.5 / 3.5    # trading cards are about 5:7 or 2.5 x 3.5 inches
TOLERANCE = 0.15             # allows for some size differences or tilt
ALPHA = 0.15                # low weight on each new box to reduce jitter
STABLE_CONFIRM_FRAMES = 10   # consecutive detections needed to trust a box
MAX_MISSED_FRAMES = 15       # keep a confirmed box through brief detection gaps
CAPTURE_MAX_MISSED_FRAMES = 3

def main():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    previous_box = None
    missed_frames = 0
    stable_frames = 0

    cnn_prediction = "None"
    vit_prediction = "None"
    mobilevit_prediction = "None"
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(12, 12))

    while True:
        ret, frame = cam.read()
        if not ret: break

        # save frame to display later
        frame = cv.flip(frame, 1)
        display_frame = frame.copy()

        # frame preprocessing
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
        blur = cv.GaussianBlur(enhanced, (5, 5), 0)
        thresh = cv.adaptiveThreshold(blur, 
                                     255, 
                                     cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv.THRESH_BINARY_INV, 
                                     blockSize=21,
                                     C=8)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        frame_area = frame.shape[0] * frame.shape[1]
        min_size = frame_area * 0.1
        max_size = frame_area * 0.6

        best_rect = None
        best_score = 0
        previous_center = None
        previous_area = None
        if previous_box is not None:
            previous_center = np.mean(previous_box, axis=0)
            previous_area = cv.contourArea(previous_box.astype(np.float32))

        for contour in contours:
            rect = cv.minAreaRect(contour)
            (cx, cy), (w, h), _ = rect

            if w == 0 or h == 0:
                continue

            area = w * h
            if area < min_size or area > max_size:
                continue

            aspect_ratio = min(w, h) / max(w, h)
            if abs(aspect_ratio - ASPECT_RATIO) >= TOLERANCE:
                continue

            # prefer upright bounding boxes
            box = cv.boxPoints(rect)
            edges = np.roll(box, -1, axis=0) - box
            edge_lengths = np.linalg.norm(edges, axis=1)
            longest_edge = edges[np.argmax(edge_lengths)]
            edge_angle = np.degrees(np.arctan2(longest_edge[1], longest_edge[0])) % 180
            vertical_deviation = abs(edge_angle - 90)
            orientation_weight = np.exp(-0.5 * (vertical_deviation / 25) ** 2)
            score = area * orientation_weight

            # reduce large movement of box
            if previous_center is not None and previous_area > 0:
                distance = np.linalg.norm(np.array([cx, cy]) - previous_center)
                movement_scale = max(np.sqrt(previous_area), 1)
                if distance > 0.35 * movement_scale:
                    continue
                score *= np.exp(-0.5 * (distance / (0.2 * movement_scale)) ** 2)

            if score > best_score:
                best_rect = rect
                best_score = score

        if best_rect is not None:
            box = cv.boxPoints(best_rect).astype(np.float32)

            if previous_box is None:
                previous_box = box
                stable_frames = 1
            else:
                center_distance = np.linalg.norm(np.mean(box, axis=0) - np.mean(previous_box, axis=0))
                previous_area = cv.contourArea(previous_box.astype(np.float32))
                consistency_limit = 0.08 * max(np.sqrt(previous_area), 1)

                # Before locking on, restart confirmation if the candidate
                # moves too much from the previous frame's candidate.
                if stable_frames < STABLE_CONFIRM_FRAMES and center_distance > consistency_limit:
                    previous_box = box
                    stable_frames = 1
                else:
                    previous_box = ALPHA * box + (1 - ALPHA) * previous_box
                    stable_frames = min(stable_frames + 1, STABLE_CONFIRM_FRAMES)

            missed_frames = 0
        else:
            missed_frames += 1
            if stable_frames < STABLE_CONFIRM_FRAMES:
                # A pending detection must be consecutive to become stable.
                previous_box = None
                stable_frames = 0
            if missed_frames >= MAX_MISSED_FRAMES:
                previous_box = None
                stable_frames = 0

        if previous_box is not None:
            if stable_frames >= STABLE_CONFIRM_FRAMES:
                color = (0, 255, 0) if missed_frames <= CAPTURE_MAX_MISSED_FRAMES else (0, 191, 255)
            else:
                color = (0, 191, 255)
            cv.drawContours(display_frame, [previous_box.astype(np.int32)], 0, color, 2)
        
        key = cv.waitKey(1) & 0xFF

        # Press Q to quit
        if key == ord('q'):
            break

        # Press Space to capture card
        elif key == ord(' '):
            if (previous_box is None
                    or stable_frames < STABLE_CONFIRM_FRAMES
                    or missed_frames > CAPTURE_MAX_MISSED_FRAMES):
                print("No stable, recent card detection. Hold the card steady and try again.")
                continue

            print("Captured!")

            min_x, min_y = np.min(previous_box, axis=0)
            max_x, max_y = np.max(previous_box, axis=0)
            x = int(min_x)
            y = int(min_y)
            width = int(max_x - min_x)
            height = int(max_y - min_y)

            card_box = gray[y:y+height, x:x+width]

            flipped = cv.flip(card_box, 1)
            cv.imshow("card crop", flipped)

            # models expect shape (540, 740)
            resized = cv.resize(flipped, (cf.SIZE_PADDED_WIDTH, cf.SIZE_PADDED_HEIGHT))
            normed = cv.normalize(resized, dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

            # Get card prediction
            predictions_cnn, predictions_vit, predictions_mobilevit = classifier2.classify(np.array(normed))

            cnn_prediction = cf.card_type_map[np.argmax(predictions_cnn[0])]
            vit_prediction = cf.card_type_map[np.argmax(predictions_vit[0])]
            mobilevit_prediction = cf.card_type_map[np.argmax(predictions_mobilevit[0])]

        cv.imshow("testing frame", thresh)

        cv.putText(display_frame, f"CNN: {cnn_prediction}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv.putText(display_frame, f"ViT: {vit_prediction}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv.putText(display_frame, f"MobileViT: {mobilevit_prediction}", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        cv.imshow('Live Demo: Trading Card Classifier', display_frame)

    cam.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()

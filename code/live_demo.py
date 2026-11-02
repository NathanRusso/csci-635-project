import cv2 as cv
import numpy as np

ASPECT_RATIO = 2.5 / 3.5    # trading cards are about 5:7 or 2.5 x 3.5 inches
TOLERANCE = 0.1             # allows for some size differences or tilt
ALPHA = 0.7                 # box smoothing parameter

def main():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    previous_box = None

    while True:
        ret, frame = cam.read()
        if not ret: break

        # save frame to display later
        frame = cv.flip(frame, 1)
        display_frame = frame.copy()

        # frame preprocessing
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        # _, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thresh = cv.adaptiveThreshold(blur, 
                                     255, 
                                     cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv.THRESH_BINARY_INV, 
                                     blockSize=21,
                                     C=8)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        # thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        
        # edges = cv.Canny(blur, 50, 150)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        # edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        min_size = frame_area * 0.1
        max_size = frame_area * 0.6
        for contour in contours:
            rect = cv.minAreaRect(contour)
            _, (w, h), _ = rect
            area = w * h

            # check size of rectangle
            if area < min_size or area > max_size:
                continue

            # check aspect ratio (about 5:7)
            aspect_ratio = min(w, h) / max(w, h)
            if np.abs(aspect_ratio - ASPECT_RATIO) >= TOLERANCE:
                continue

            # draw bounding box
            box = cv.boxPoints(rect)
            box = np.int64(box)

            if previous_box is None:
                previous_box = box.astype(np.float32)
            else:
                previous_box = (
                    ALPHA * box.astype(np.float32)
                    + (1 - ALPHA) * previous_box
                )

            smoothed_box = previous_box.astype(np.int32)

            cv.drawContours(display_frame, [smoothed_box], 0, (0, 255, 0), 2)

        cv.imshow("testing frame", thresh)
        cv.imshow('Live Demo: Trading Card Classifier', display_frame)
        
        # Press Q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
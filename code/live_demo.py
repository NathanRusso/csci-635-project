import cv2 as cv
import numpy as np
from tensorflow import keras

ASPECT_RATIO = 2.5 / 3.5    # trading cards are about 5:7 or 2.5 x 3.5 inches
TOLERANCE = 0.15             # allows for some size differences or tilt
ALPHA = 0.7                 # box smoothing parameter

class_names = {
    0: "Magic: The Gathering",
    1: "Pokemon",
    2: "Yu-Gi-Oh"
}

def main():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    previous_box = None

    # using the CNN model
    model = keras.models.load_model("data/635-CNN.keras")

    # # using the Mobile ViT model
    # # got a ton of warnings/errors, not sure why
    # model = keras.models.load_model("data/635-MobileVit_v2.keras")

    print(model.summary())

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

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        # thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        
        # edges = cv.Canny(blur, 100, 200)
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        # thresh = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        min_size = frame_area * 0.1
        max_size = frame_area * 0.6
        min_angle = -110
        max_angle = -80
        for contour in contours:
            rect = cv.minAreaRect(contour)
            _, (w, h), angle = rect
            
            if angle < min_angle or angle > max_angle:
                continue
            
            # check size of rectangle
            area = w * h
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
                previous_box = (ALPHA * box.astype(np.float32) + (1 - ALPHA) * previous_box)

            smoothed_box = previous_box.astype(np.int32)

            cv.drawContours(display_frame, [smoothed_box], 0, (0, 255, 0), 2)

        cv.imshow("testing frame", thresh)
        cv.imshow('Live Demo: Trading Card Classifier', display_frame)
        
        key = cv.waitKey(1) & 0xFF

        # Press Q to quit
        if key == ord('q'):
            break

        # Press Space to capture card
        elif key == ord(' '):
            if previous_box is None:
                print("No card found!")
                continue

            print("Captured!")

            min_x, min_y = np.min(previous_box, axis=0)
            max_x, max_y = np.max(previous_box, axis=0)
            x = int(min_x)
            y = int(min_y)
            width = int(max_x - min_x)
            height = int(max_y - min_y)

            card_box = gray[y:y+height, x:x+width]

            cv.imshow("card crop", card_box)

            # CNN expects shape (None, 700, 500, 1)
            resized = cv.resize(card_box, (500, 700))
            normed = cv.normalize(resized, dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

            normed = np.expand_dims(normed, axis=-1)
            input_image = np.expand_dims(normed, axis=0)

            # Get card prediction
            prediction = model.predict(input_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            predicted_label = class_names[predicted_class]

            # Get confidence
            confidence = prediction[0][predicted_class]

            print(f"Prediction: {predicted_label}")
            print(f"Confidence: {confidence * 100:.2f}%")


    cam.release()
    cv.destroyAllWindows()





if __name__ == "__main__":
    main()
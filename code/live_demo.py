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
import classifier1
# from keras_vision.MobileViT_v2.base_layers import ConvLayer, InvertedResidualBlock
# from keras_vision.MobileViT_v2.mobile_vit_v2_block import MobileViT_v2_Block

ASPECT_RATIO = 2.5 / 3.5    # trading cards are about 5:7 or 2.5 x 3.5 inches
TOLERANCE = 0.15             # allows for some size differences or tilt
ALPHA = 0.7                 # box smoothing parameter


def main():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    previous_box = None
    cnn_prediction = "None"
    vit_prediction = "None"
    mobilevit_prediction = "None"

    # using the CNN model
    # model = keras.models.load_model("data/635-CNN.keras")

    # using the Mobile ViT model
    # got a ton of warnings/errors, not sure why
    # model = keras.models.load_model(
    #     "data/635-MobileVit_v2.keras",
    #     custom_objects={
    #         "ConvLayer": ConvLayer,
    #         "InvertedResidualBlock": InvertedResidualBlock,
    #         "MobileViT_v2_Block": MobileViT_v2_Block,
    #     },
    # )

    # # using the ViT model
    # model = keras.models.load_model("data/635-ViT.keras")

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

            flipped = cv.flip(card_box, 1)
            cv.imshow("card crop", flipped)

            # CNN expects shape (None, 700, 500, 1)
            resized = cv.resize(flipped, (cf.SIZE_FEATURE_WIDTH, cf.SIZE_FEATURE_HEIGHT))
            normed = cv.normalize(resized, dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

            # Get card prediction
            # prediction = model.predict(input_image, verbose=0) 
            # prediction = classifier.vit_classify(np.array(input_image))
            predictions_cnn, predictions_vit, predictions_mobilevit = classifier1.classify(np.array(normed))

            # predicted_class = np.argmax(prediction[0])
            # print(prediction[0])
            # predicted_label = cf.card_type_map[predicted_class]
            cnn_prediction = cf.card_type_map[np.argmax(predictions_cnn[0])]
            vit_prediction = cf.card_type_map[np.argmax(predictions_vit[0])]
            mobilevit_prediction = cf.card_type_map[np.argmax(predictions_mobilevit[0])]

            # # Get confidence
            # confidence = prediction[0][predicted_class]

            # print(f"Prediction: {predicted_label}")
            # print(f"Confidence: {confidence * 100:.2f}%")
            # classifier.display_classify(np.array(input_image), ["Live Demo"])

        cv.imshow("testing frame", thresh)

        cv.putText(display_frame, f"CNN: {cnn_prediction}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv.putText(display_frame, f"ViT: {vit_prediction}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv.putText(display_frame, f"MobileViT: {mobilevit_prediction}", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        cv.imshow('Live Demo: Trading Card Classifier', display_frame)

    cam.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
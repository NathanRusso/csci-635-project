import cv2 as cv
import numpy as np

import config as cf
import classifier2


ASPECT_RATIO = 2.5 / 3.5
BOX_SCALE = 0.85


def get_fixed_box(frame_shape):
    """Return a card-shaped box centered in the rightmost third of the frame."""
    frame_height, frame_width = frame_shape[:2]
    third_width = frame_width / 2.5

    box_height = int(min(frame_height * BOX_SCALE,
                         third_width * BOX_SCALE / ASPECT_RATIO))
    box_width = int(box_height * ASPECT_RATIO)

    center_x = int(frame_width * 2 / 3)
    center_y = frame_height // 2
    x1 = center_x - box_width // 2
    y1 = center_y - box_height // 2
    return x1, y1, x1 + box_width, y1 + box_height


def main():
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    cnn_prediction = "None"
    vit_prediction = "None"
    mobilevit_prediction = "None"

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Mirror the preview so it behaves like a mirror.
            frame = cv.flip(frame, 1)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            display_frame = frame.copy()

            x1, y1, x2, y2 = get_fixed_box(frame.shape)
            cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(display_frame, "Align card here | SPACE: capture | Q: quit",
                       (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv.putText(display_frame, f"CNN: {cnn_prediction}", (20, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv.putText(display_frame, f"ViT: {vit_prediction}", (20, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv.putText(display_frame, f"MobileViT: {mobilevit_prediction}", (20, 145),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv.imshow("Live Demo: Trading Card Classifier", display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # Capture the fixed box, without attempting to detect card edges.
                card_crop = gray[y1:y2, x1:x2]
                if card_crop.size == 0:
                    print("Error: The fixed capture box is outside the camera frame.")
                    continue

                # The preview is mirrored, so unmirror the crop for classification.
                card_crop = cv.flip(card_crop, 1)
                cv.imshow("Captured card crop", card_crop)

                resized = cv.resize(card_crop, (cf.SIZE_PADDED_WIDTH, cf.SIZE_PADDED_HEIGHT))
                normed = cv.normalize(resized, dst=None, alpha=0.0, beta=1.0, 
                                      norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

                predictions_cnn, predictions_vit, predictions_mobilevit = classifier2.classify(
                    np.array(normed)
                )
                cnn_prediction = cf.card_type_map[np.argmax(predictions_cnn[0])]
                vit_prediction = cf.card_type_map[np.argmax(predictions_vit[0])]
                mobilevit_prediction = cf.card_type_map[np.argmax(predictions_mobilevit[0])]

    finally:
        cam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()

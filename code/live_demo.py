import cv2 as cv



def main():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # frame preprocessing
        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.GaussianBlur(frame, (5, 5), 0)
        # _, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        frame = cv.adaptiveThreshold(frame, 
                                     255, 
                                     cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv.THRESH_BINARY_INV, 
                                     blockSize=11,
                                     C=4)

        
        cv.imshow('Live Demo: Trading Card Classifier', frame)
        
        # Press Q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
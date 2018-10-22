import face_recognition
import cv2

def getWebcam():
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


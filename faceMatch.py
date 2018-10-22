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

def getFace(file):
    rawFace = face_recognition.load_image_file(file)
    return face_recognition.face_encodings(rawFace)[0]

#me = getFace("myface.png")
# notMe = getFace("notme.jpg")
# identify = getFace("whoseface.JPG")
# results = face_recognition.compare_faces([me,notMe], identify)
# print(results)



face_locations = face_recognition.face_locations(face_recognition.load_image_file("myface.png"))
print(face_locations)
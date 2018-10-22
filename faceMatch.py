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


# getWebcam()


face = cv2.imread("myface.png")

cv2.imshow('image',img)
cv2.imshow('title', face)



#me = getFace("myface.png")
# notMe = getFace("notme.jpg")
# identify = getFace("whoseface.JPG")
# results = face_recognition.compare_faces([me,notMe], identify)
# print(results)


#
# # top, right, bottom, left
# fL = face_recognition.face_locations(face_recognition.load_image_file("myface.png"))
# print(fL[0][0])
# tL = (fL[0][3], fL[0][0])
# bR = (fL[0][1], fL[0][2])
# cv2.rectangle(face,tL,bR, (0, 0, 255), 2)
# cv2.imshow("yah",face)




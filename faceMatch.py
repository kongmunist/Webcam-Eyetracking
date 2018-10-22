import face_recognition
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw


def getWebcam():
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()

        locations = face_recognition.face_locations(frame)

        for spot in locations:
            tL, bR = getCorners(spot)
            cv2.rectangle(frame,tL,bR,(0,0,255),3)





        cv2.imshow('frame', frame)





        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

def getFace(file):
    rawFace = face_recognition.load_image_file(file)
    return face_recognition.face_encodings(rawFace)[0]

def getCorners(fL):
    tL = (fL[3], fL[0])
    bR = (fL[1], fL[2])
    return tL, bR

# getWebcam()

face = face_recognition.load_image_file("notme.jpg")
img = Image.fromarray(face)
d = ImageDraw.Draw(img, 'RGBA')
feats = face_recognition.face_landmarks(face_recognition.load_image_file("notme.jpg"))[0]
d.polygon(feats['left_eye'], fill=(0, 0, 255, 255))

img.show()





#me = getFace("myface.png")
# notMe = getFace("notme.jpg")
# identify = getFace("whoseface.JPG")
# results = face_recognition.compare_faces([me,notMe], identify)
# print(results)



# # top, right, bottom, left
# fL = face_recognition.face_locations(face_recognition.load_image_file("myface.png"))
# print(fL[0][0])
# tL = (fL[0][3], fL[0][0])
# bR = (fL[0][1], fL[0][2])
# cv2.rectangle(face,tL,bR, (0, 0, 255), 2)
# cv2.imshow("yah",face)
# cv2.waitKey(0)


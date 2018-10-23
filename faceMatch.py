import face_recognition
import numpy as np
import cv2
import copy
from PIL import Image
from PIL import ImageDraw


def getWebcam():
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        lowFiFrame = copy.deepcopy(frame)
        lowFiFrame = cv2.resize(lowFiFrame, (0,0), fy=.25, fx=.25)
        locations = face_recognition.face_locations(lowFiFrame)

        for spot in locations:
            tL, bR = getCorners(spot,4)
            cv2.rectangle(frame,tL,bR,(0,0,255),3)
            try:
                feats = face_recognition.face_landmarks(lowFiFrame)
                for person in feats:
                    mark = person["left_eye"]
                    mark = np.multiply(np.array(mark, np.int32).reshape((-1,1,2)),4)
                    print(mark)
                    cv2.polylines(frame, [mark],True,(0,255,0))
                # # feats = np.array(feats,np.int32).reshape((-1,1,2))
                # newFeats = []
                # for tup in feats:
                #     newFeats.append([tup[0], tup[1]])
                # newFeats = np.multiply(
                #     np.array(newFeats, np.int32).reshape((-1, 1, 2)), 4)
                # cv2.polylines(frame, [newFeats], True, (0, 255, 0))
            except:
                pass



        cv2.imshow('frame', frame)





        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

def getFace(file):
    rawFace = face_recognition.load_image_file(file)
    return face_recognition.face_encodings(rawFace)[0]

def getCorners(fL,n):
    tL = (fL[3]*n, fL[0]*n)
    bR = (fL[1]*n, fL[2]*n)
    return tL, bR

def makeFeatRelative(feature):
    sumX = 0
    sumY = 0
    for coord in feature:
        sumX += coord[0]
        sumY += coord[1]
    avgX = sumX // len(feature)
    avgY = sumY // len(feature)
    relativeArray = []
    for coord in feature:
        relativeArray.append((coord[0]-avgX,coord[1]-avgY))
    return (avgX,avgY), relativeArray

getWebcam()

#
# face = face_recognition.load_image_file("jim.png")
# img = Image.fromarray(face)
# d = ImageDraw.Draw(img, 'RGBA')
# feats = face_recognition.face_landmarks(face_recognition.load_image_file("jim.png"))[0]
# d.polygon(feats['left_eye'], fill=(0, 0, 255, 255))
# d.polygon(feats["nose_tip"],fill = (255,0,0,255))
# d.polygon(feats["nose_bridge"],fill = (255,128,0,255))
# d.polygon(feats["bottom_lip"],fill = (123,128,0,255))
# d.polygon(feats["top_lip"],fill = (123,128,0,255))
# d.polygon(feats["left_eyebrow"],fill = (0,128,100,255))
#
# img.show()





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


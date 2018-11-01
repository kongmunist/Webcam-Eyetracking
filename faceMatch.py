import face_recognition
import numpy as np
import cv2
import copy
from PIL import Image
from PIL import ImageDraw


def getWebcam():
    webcam = cv2.VideoCapture(0)
    #Frame coordinates go frame[y][x]
    while True:
        ret, frame = webcam.read()
        lowFiFrame = cv2.resize(copy.deepcopy(frame), (0,0), fy=.25, fx=.25)
        locations = face_recognition.face_locations(lowFiFrame)
        feats = face_recognition.face_landmarks(lowFiFrame)

        tagFaces(frame,locations)
        featureSwap(frame, feats, "left_eye", "right_eye")

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


#Draws red box around faces it sees
def tagFaces(frame, locations):
    for spot in locations:
        tL, bR = getCorners(spot, 4)
        cv2.rectangle(frame, tL, bR, (0, 0, 255), 3)

def featureSwap(frame, feats, feat1,feat2):
    if len(feats) == 0:
        return False
    eyeList = []
    for person in feats:
        eyeList.append(person[feat1])
        eyeList.append(person[feat2])
    eyeRectList = []
    for eye in eyeList:
        eyeRectList.append(maxAndMin(eye))
    for i in range(len(eyeList)//2):
        eyeSwap(frame, eyeRectList[i], eyeRectList[i+1])

def maxAndMin(featCoords):
    adj = 5
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    return [min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj]

def eyeSwap(frame, eye1, eye2):
    # Coords in the eyes go minx, miny, maxx, maxy
    try:
        for x in range((eye1[2]-eye1[0]) * 4):
            for y in range((eye1[3]-eye1[1]) * 4):
                firstEye = copy.copy(frame[eye2[1]*4 + y][eye2[0]*4 + x])
                frame[eye2[1]*4 + y][eye2[0]*4 + x] = frame[eye1[1]*4+y][eye1[0]*4+x]
                frame[eye1[1]*4 + y][eye1[0]*4 + x] = firstEye
    except:
        pass

def editFeat(origPic, smallPic, feats, feature, color, borderSize):
    try:
        for person in feats:
            mark = person[feature]
            mark = np.multiply(np.array(mark, np.int32).reshape((-1, 1, 2)), 4)
            cv2.polylines(origPic, [mark], True, color, borderSize)
    except:
        pass

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


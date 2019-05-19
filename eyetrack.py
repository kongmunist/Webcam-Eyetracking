import face_recognition
import numpy as np
import cv2 as cv
import copy
from matplotlib import pyplot as plt
import pyautogui
import time
import os


# pyautogui.moveRel(0, 10) # move mouse 10 pixels down
# pyautogui.dragTo(100, 150)
# pyautogui.dragRel(0, 10) # drag mouse 10 pixels down
# pyautogui.scroll(200)

# detector.filterByArea = True
# detector.blobColor = 0

def maxAndMin(featCoords,mult = 1):
    adj = 10/mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj])
    print(maxminList)
    return (maxminList*mult).astype(int), (np.array([sum(listX)/len(listX)-maxminList[0], sum(listY)/len(listY)-maxminList[1]])*mult).astype(int)

def findCircs(img):
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, 20, param1 = 200, param2 = 50, minRadius=1, maxRadius=40)#, minRadius = 0, maxRadius = 30)
    # circles = np.uint16(np.around(circles))
    return circles

def findBlobs(img):
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200

    # params.filterByColor = True
    # params.blobColor = 0

    params.filterByArea = True
    params.maxArea = 3000

    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    # imkeypoints = cv.drawKeypoints(img, keypoints, np.array([]),
    #                                (0, 0, 255),
    #                                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints





def getWebcam(feed=False):
    webcam = cv.VideoCapture(0)
    # Frame coordinates go frame[y][x]
    haventfoundeye = True
    screenw = 1440
    screenh = 900

    while True:
        ret, frame = webcam.read()
        smallframe = cv.resize(copy.deepcopy(frame), (0,0), fy=.15, fx=.15)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:

            leBds,leCenter = maxAndMin(feats[0]['left_eye'],mult = 1/.15)
            # reBds,_ = maxAndMin(feats[0]['right_eye'])
            # print(leBds)

            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            # right_eye = frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]

            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(left_eye, 50, 255, 0)

            # Find weighted average for center of the eye
            TMP = 255 - np.copy(thresh)#.astype(int)
            # TMP = TMP[0:-1, 10:-10]
            # cv.imshow("tmp", TMP)
            # TMP = cv.blur(TMP, (3, 3))
            y = np.sum(TMP, axis=1)
            x = np.sum(TMP, axis=0)
            # x = TMP[int(len(TMP)/2)]
            y = y / len(TMP[0])
            x = x / len(TMP)

            y = y > np.average(y) + np.std(y)#*1.2
            x = x > np.average(x) + np.std(x)#*1.2

            try:
                y = int(np.dot(np.arange(1, len(y) + 1), y) / sum(y))
            except:
                y = int(np.dot(np.arange(1, len(y) + 1), y) / 1)

            try:
                x = int(np.dot(np.arange(1, len(x) + 1), x) / sum(x))
            except:
                x = int(np.dot(np.arange(1, len(x) + 1), x) / 1)

            haventfoundeye = False


            left_eye = cv.cvtColor(left_eye, cv.COLOR_GRAY2BGR)
            cv.circle(left_eye, (x, y), 2, (20, 20, 120), 3)
            cv.circle(left_eye, (int(leCenter[0]), int(leCenter[1])), 2, (120, 20, 20), 3)

            # screenx = screenw/2 + ((leCenter[0]-x))/20*screenw
            # screeny = screenh/2 + ((y-leCenter[1])+5.8)/10*screenh

            # print(leCenter[0]-x, y-leCenter[1])
            # print(screenx,screeny)
            # pyautogui.moveTo(screenx,screeny)



            if feed:
                cv.imshow('frame', left_eye)



                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            elif not haventfoundeye:
                plt.imshow(left_eye)
                plt.title('my EYEBALL')
                plt.show()
                return left_eye
        
        # # Range of sizes is about 55x100 to 85x160, rescale to like
# pyautogui.FAILSAFE = False


def getEye(times = 1,frameShrink = 0.15, coords = (0,0), counterStart = 0, folder = "eyes"):
    os.makedirs(folder, exist_ok=True)
    webcam = cv.VideoCapture(0)
    counter = counterStart
    ims = []

    while counter < counterStart+times:
        ret, frame = webcam.read()
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)

            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            # right_eye = frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]

            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)

            left_eye = cv.resize(left_eye, dsize=(100, 50))

            # D
            # isplay the image - DEBUGGING ONLY
            cv.imshow('frame', left_eye)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            cv.imwrite(
                folder + "/" + str(coords[0]) + "." + str(coords[1]) + "." + str(
                    counter) + ".jpg", left_eye)
            counter += 1


# # 1440x900
# for i in [0,720,1440]:
#     for j in [0,450,900]:
for i in [404,951]:
    for j in [383,767]:
        print(i,j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times = 10, coords=(i,j),counterStart=0, folder = "testeyes")

# getEye(times = 1, coords=(360,225),counterStart=0)
# getWebcam(True)
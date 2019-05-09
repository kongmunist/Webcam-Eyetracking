import face_recognition
import numpy as np
import cv2 as cv
import copy
from matplotlib import pyplot as plt
import pyautogui


# pyautogui.moveRel(0, 10) # move mouse 10 pixels down
# pyautogui.dragTo(100, 150)
# pyautogui.dragRel(0, 10) # drag mouse 10 pixels down
# pyautogui.scroll(200)

# detector.filterByArea = True
# detector.blobColor = 0

def maxAndMin(featCoords):
    adj = 10
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = [min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj]
    return maxminList, [sum(listX)/len(listX)-maxminList[0], sum(listY)/len(listY)-maxminList[1]]

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
        # frame = cv.resize(copy.deepcopy(frame), (0,0), fy=.25, fx=.25)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        feats = face_recognition.face_landmarks(frame)
        if len(feats) > 0:
            leBds,leCenter = maxAndMin(feats[0]['left_eye'])
            reBds,_ = maxAndMin(feats[0]['right_eye'])

            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            right_eye = frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]


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

            #             print(y)
            #             print(len(y))
            #             y = np.array([y[i]*i for i in range(len(y))])
            #             x = np.array([x[i]*i for i in range(len(y))])

            #             y = int(np.sum(y)/255)
            #             x = int(np.sum(x)/255)

                        # circles = findCircs(left_eye)
            #             try:
            #                 if ([0,0,0] not in circles[0]):
            #                     print(circles)
            #                     circles = circles[0]
            #                     left_eye = cv.cvtColor(left_eye, cv.COLOR_GRAY2BGR)
            #                     for i in circles:
            #                         cv.circle(left_eye, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # #                         cv.circle(left_eye, (i[0], i[1]), 2, (0, 0, 255), 2)
            #                         haventfoundeye = False
            #             except:
            #                 pass

            # circles = findCircs(left_eye)
            # blobs = findBlobs(left_eye)
            # try:
            #     print(blobs[0].pt[0])
            #     print(blobs[0].pt[1])
            # except:
            #     print(blobs)
            # left_eye = cv.drawKeypoints(left_eye, blobs, np.array([]),
            #                                (0, 0, 255),
            #                                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # # im2, c, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
            # #                                     cv.CHAIN_APPROX_SIMPLE)
            # #
            left_eye = cv.cvtColor(left_eye, cv.COLOR_GRAY2BGR)

            # #
            # # cv.drawContours(left_eye, c,-1, (120,120,12))
            # #
            # # cv.imshow("thresh",thresh)
            #
            # try:
            #     for pt in feats[0]['left_eye']:
            #         cv.circle(left_eye, (pt[1] - leBds[1], pt[0] - leBds[0]), 3)
            #         pass
            # except:
            #     pass


            center = [len(left_eye[0])/2,len(left_eye)/2]


            relx = x-center[0]
            rely = y-center[1]

            cv.circle(left_eye, (x, y), 2, (20, 20, 120), 3)
            cv.circle(left_eye, (int(leCenter[0]), int(leCenter[1])), 2, (120, 20, 20), 3)

            screenx = screenw/2 + ((leCenter[0]-x))/20*screenw
            screeny = screenh/2 + ((y-leCenter[1])+5.8)/10*screenh

            # print(leCenter[0]-x, y-leCenter[1])
            print(screenx,screeny)
            pyautogui.moveTo(screenx,screeny)
            # print(((leCenter[0]-x)+0.5)/17.5,((y-leCenter[1])+5.1)/4)




            # -9.2, -7        8.3, -3.2



            # screenx = (1/0.16439)*relx/(len(left_eye[0]))*screenw
            # screenx = screenw/2 - (100/0.16439)*relx/(len(left_eye[0]))*screenw
            # screeny = screenh/2 + (100/0.1318681)*rely/(len(left_eye))*screenh
            # screeny = (1/0.1318681)*rely/(len(left_eye))*screenh
            # print(screenx, screeny)
            # print()
            #
            # print(relx/(len(left_eye[0])))
            # # print(relx/(len(left_eye[0])*.2))
            # # print(rely/(len(left_eye)*.15))
            # print(rely/(len(left_eye)))
            # print()

            # TL
            # 0.1043956043956044
            # -0.0690476190476191
            #
            # BR
            # -0.07894736842105263
            # 0.0628205128205128

            if feed:
                cv.imshow('frame', left_eye)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            elif not haventfoundeye:
                plt.imshow(left_eye)
                plt.title('my EYEBALL')
                plt.show()
                return left_eye
        #
        # # Range of sizes is about 55x100 to 85x160, rescale to like
pyautogui.FAILSAFE = False

getWebcam(True)
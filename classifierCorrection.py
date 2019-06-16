import os
import cv2 as cv
import face_recognition
import numpy as np
import copy
import torch
import torch.nn as nn
import MLtracking as ml
import pyautogui
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# To make examples:
# 1. take eye pic
# 2. run models on it
# 3. record their predictions for N cycles
# 4. write down the N guesses and the actual looking position




# # X classifiers
# sixn = ml.sixnine().to(device)
# sixn.load_state_dict(torch.load("xModels/69good.plt",map_location=device))
# sixn.eval()
#
# sevent = ml.venty().to(device)
# sevent.load_state_dict(torch.load("xModels/70test.plt",map_location=device))
# sevent.eval()
#
#
# # Y classifiers
# fiv = ml.eightdeep().to(device)
# fiv.load_state_dict(torch.load("yModels/54x1.plt",map_location=device))
# fiv.eval()
#
#
#
# def ensembleX(im): # 58 accuracy
#     modList = [sixn,sevent]
#     sumn = 0
#     for mod in modList:
#         sumn += mod(im).item()
#     return sumn / len(modList)


def maxAndMin(featCoords,mult = 1):
    adj = 10/mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj])
    # print(maxminList)
    return (maxminList*mult).astype(int), (np.array([sum(listX)/len(listX)-maxminList[0], sum(listY)/len(listY)-maxminList[1]])*mult).astype(int)


#Preps a color pic of the eye for input into the CNN
def process(im):
    left_eye = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    left_eye = cv.resize(left_eye, dsize=(100, 50))

    # Display the image - DEBUGGING ONLY
    cv.imshow('frame', left_eye)

    top = max([max(x) for x in left_eye])
    left_eye = (torch.tensor([[left_eye]]).to(dtype=torch.float,
                                              device=device)) / top
    return left_eye


#
# # def makeExamples():
# webcam = cv.VideoCapture(0)
# frameShrink = 0.15
# mvAvgx = []
# mvAvgy = []
# scale = 10
# margin = 200
#
# pixelLocation = 10
# pixelRecord = []
# xRecord = []
# err = []
# counter = 0

# while pixelLocation < 1400:
#     ret, frame = webcam.read()
#     smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink,
#                            fx=frameShrink)
#     smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)
#
#     feats = face_recognition.face_landmarks(smallframe)
#
#
#
#     if len(feats) > 0:
#
#         leBds, leCenter = maxAndMin(feats[0]['left_eye'],
#                                     mult=1 / frameShrink)
#         left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
#
#         left_eye = process(left_eye)
#
#         x = ensembleX(left_eye) * 1440
#         y = fiv(left_eye).item() * 900
#         pyautogui.moveTo(pixelLocation,450)
#
#
#
#         pixelRecord.append(pixelLocation)
#         xRecord.append(x)
#         err.append(abs(x-pixelLocation))
#         counter += 1
#
#         if counter % 5 == 0:
#             pixelLocation += 50
#
#
#
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

def eyetrack(frameShrink = 0.15):

    webcam = cv.VideoCapture(0)

    o = open("corrections.txt","r+")

    for i in range(40,1440,1440//10):
        for j in range(0,900,900//10):
            o.write(str(i/1440) + "," + str(j/900))
            print(i,j)
            pyautogui.moveTo(i,j)
            for k in range(10):

                ret, frame = webcam.read()
                smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
                smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

                feats = face_recognition.face_landmarks(smallframe)
                if len(feats) > 0:
                    leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
                    left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]

                    left_eye = process(left_eye)


                    x = ensembleX(left_eye)
                    y = fiv(left_eye).item()


                    o.write("," + str(x) + "," + str(y))


                    if cv.waitKey(1) & 0xFF == ord('q'):
                        o.close()
                        break
            o.write("\n")
    o.close()
# eyetrack()


def benchmark(set):
    offx = 0
    offy = 0
    for thing in set:
        offset = abs(thing[0] - fick(thing[2:4]))
        offx += offset[0]
        # offy += offset[1]
    return offx/len(set)*1440#,offy/len(set)*900



class lin(torch.nn.Module):
    def __init__(self):
        super(lin, self).__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        # self.linear2 = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.linear2(x)

        return x



filename = 'corrections.txt'
fick = lin()
criterion = torch.nn.MSELoss(size_average = False)
# optimizer = torch.optim.SGD(fick.parameters(), lr = 0.01)
optimizer = torch.optim.Adam(fick.parameters(), lr=0.001)

numEpochs = 50

# def loadData(filename):
labels = []
datum = []
f = open(filename, "r")
test = [x.strip().split(",") for x in f.readlines()]
test = [[float(x) for x in y] for y in test]


train = []
for thing in test:
    while len(thing) > 4:
        train.append([thing[0],thing[1],thing.pop(2),thing.pop(2)])

test = torch.tensor(test)
train = torch.tensor(train)


for i in range(numEpochs):
    print(i)
    for exm in train:

        pred = fick.forward(exm[2:4])
        loss = criterion(pred, exm[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i%10 == 0:
        fick.eval()
        print(benchmark(train))
        print(benchmark(test))
        fick.train()




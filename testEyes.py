import os
import cv2 as cv
import face_recognition
import numpy as np
import copy
import torch
import torch.nn as nn
import torchvision
import torch.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        f1 = 4
        f2 = 16
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(25 * 12 * f2, 400)
        self.fc2 = nn.Linear(400, 60)
        self.fc3 = nn.Linear(60, 1)



    def forward(self,x):
        x = self.layer1(x);
        x = self.layer2(x);
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x);
        x = self.fc2(x);
        x = self.fc3(x);

        return x

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

def getEye(model, times = 1,frameShrink = 0.15, coords = (0,0), counterStart = 0, folder = "eyes"):
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
            pred = model(torch.tensor([[left_eye]],dtype=torch.float))
            print(1440*pred.item())

            if cv.waitKey(1) & 0xFF == ord('q'):
                break



mod = ConvNet()
mod.load_state_dict(torch.load("xModels/96x.plt"))
mod.eval()
getEye(mod)
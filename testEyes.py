import os
import cv2 as cv
import face_recognition
import numpy as np
import copy
import torch
import torch.nn as nn
import torchvision
import torch.functional as F

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device  = 'cpu'

class eightdeep(torch.nn.Module):
    def __init__(self):
        super(eightdeep, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(f2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 1)



    def forward(self,x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class fourdeep(torch.nn.Module):
    def __init__(self):
        super(fourdeep, self).__init__()

        f2 = 4
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(f2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 1)



    def forward(self,x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class venty(torch.nn.Module):
    def __init__(self):
        super(venty, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 1)



    def forward(self,x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class seven(torch.nn.Module):
    def __init__(self):
        super(seven, self).__init__()

        f2 = 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 2000)
        self.fc2 = nn.Linear(2000, 200)
        self.fc3 = nn.Linear(200, 1)



    def forward(self,x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class eightfour(torch.nn.Module):
    def __init__(self):
        super(eightfour, self).__init__()

        f1 = 8
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


class sixnine(torch.nn.Module):
    def __init__(self):
        super(sixnine, self).__init__()

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
        self.fc3 = nn.Linear(60, 10)
        self.fc4 = nn.Linear(10, 1)



    def forward(self,x):
        x = self.layer1(x);
        x = self.layer2(x);
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x);
        x = self.fc2(x);
        x = self.fc3(x);
        x = self.fc4(x);


        return x

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

def dataLoad(path, want = 0):
    nameList = os.listdir(path)

    try:
        nameList.remove(".DS_Store")
    except:
        pass
    totalHolder = []
    dims = [1440,900]

    for name in nameList:
        im = cv.cvtColor(cv.imread(path + "/" + name), cv.COLOR_BGR2GRAY)
        top = max([max(x) for x in im])
        totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                            torch.tensor([[int((name.split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

    # print(totalHolder)
    return totalHolder


def evaluateModel(model,testSet, sidelen = 1440):
    # model.eval()
    err = 0
    for (im, label) in testSet:
        output = model(im)
        err += abs(output - label.item())
    # model.train()

    return (err/len(testSet)*sidelen)


# X classifiers

# sevsev = seven().to(device)
# sevsev.load_state_dict(torch.load("xModels/77good.plt"))
# sevsev.eval()
#
# sevn = ConvNet().to(device)
# sevn.load_state_dict(torch.load("xModels/79good.plt"))
# sevn.eval()
#
# eighfour = ConvNet().to(device)
# eighfour.load_state_dict(torch.load("xModels/84good.plt"))
# eighfour.eval()
#
# eighnine = ConvNet().to(device)
# eighnine.load_state_dict(torch.load("xModels/89good.plt"))
# eighnine.eval()
#
# eighfive = eightfour().to(device)
# eighfive.load_state_dict(torch.load("xModels/85good.plt"))
# eighfive.eval()
#
# se = fourdeep().to(device)
# se.load_state_dict(torch.load("xModels/68.plt"))
# se.eval()
#
# sn = fourdeep().to(device)
# sn.load_state_dict(torch.load("xModels/69.plt"))
# sn.eval()

sixn = sixnine().to(device)
sixn.load_state_dict(torch.load("xModels/69good.plt",map_location=device))
sixn.eval()

sevent = venty().to(device)
sevent.load_state_dict(torch.load("xModels/70test.plt",map_location=device))
sevent.eval()


# Y classifiers
fiv = eightdeep().to(device)
fiv.load_state_dict(torch.load("yModels/54x1.plt",map_location=device))
fiv.eval()

# sone = fourdeep().to(device)
# sone.load_state_dict(torch.load("yModels/61.plt"))
# sone.eval()
#
# stwo = fourdeep().to(device)
# stwo.load_state_dict(torch.load("yModels/62.plt"))
# stwo.eval()

testy = dataLoad("testeyes",want=1)
testx = dataLoad("testeyes")
print(evaluateModel(sixn, testx))

trainx = dataLoad("eyes")
trainy = dataLoad("eyes",want=1)


def ensembleX(im): # 58 accuracy
    modList = [sixn,sevent]
    sumn = 0
    for mod in modList:
        sumn += mod(im).item()
    return sumn / len(modList)


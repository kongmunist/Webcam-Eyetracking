import numpy as np
# import eyetrack as eye
import cv2 as cv
import os
import copy

import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# class noPool(torch.nn.Module):
#     def __init__(self):
#         super(noPool,self).__init__()



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
    model.eval()
    err = 0
    for (im, label) in testSet:
        output = model(im)
        err += abs(output.item() - label.item())
    model.train()

    return (err/len(testSet)*sidelen)

def getError(model,testSet):
    model.eval()
    foeList = []
    for (im,label) in testSet:
        output = model(im)
        foeList.append(abs(output.item() - label.item()))
    model.train()
    return foeList


num_epochs = 2

model = ConvNet().to(device)
# model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trainingSet = dataLoad("eyes")
test = dataLoad("testeyes")

# trainingSet = [(x.to(device),y.to(device)) for (x,y) in trainingSet]
# test = [(x.to(device),y.to(device)) for (x,y) in test]

# print(trainingSet[-1])
# print(test[-1])


bestModel = model
bestScore = 10000
testscores = []
trainscores = []

model.train()
for epoch in range(num_epochs):
    print(epoch)
    np.random.shuffle(trainingSet)

    for i,(im, label) in enumerate(trainingSet):

        # print(im.device)
        # print(label.device)


        output = model(im)
        # if label.item()<.2:
        #     print(i, output, label)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            testSc = evaluateModel(model,test)
            trainSc = evaluateModel(model,trainingSet)
            if testSc < bestScore:
                bestModel = copy.deepcopy(model)
                bestScore = testSc
            testscores.append(testSc)
            trainscores.append(trainSc)
            print(trainSc)
            print(testSc)



            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(trainingSet), loss.item()))



finalScore = evaluateModel(bestModel,test)
print(finalScore)

if finalScore < 100:
    torch.save(bestModel.state_dict(), "xModels/" + str(int(finalScore))+".plt")

plt.title(str(int(finalScore)))
plt.plot(testscores)
plt.plot(trainscores)
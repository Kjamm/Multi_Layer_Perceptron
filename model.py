import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim

trainDataSet = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testDataSet = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainLoader = DataLoader(dataset=trainDataSet, batch_size=32, shuffle=True)
testLoader = DataLoader(dataset=testDataSet, batch_size=32, shuffle=False)

class MyMLP(nn.Module) :
    def __init__(self) :
        super(MyMLP, self).__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x) :
        x = x.view(-1, 28 * 28)
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = self.linear2(x)
        x = F.sigmoid(x)
        x = self.out(x)

        return x

model = MyMLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
num_epochs = 20

for epoch in range(num_epochs) :
    runningLoss = 0.0

    for data in trainLoader :
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        runningLoss += loss.item()

    print(f"Epoch {epoch + 1}, Loss : {runningLoss / len(trainLoader):.4f}")

totalPrediction = 0
correctPrediction = 0

with torch.no_grad() :
    for data in testLoader :
        inputs, labels = data
        outputs = model(inputs)

        _, pred = torch.max(outputs, dim=1)
        totalPrediction += len(labels)

        correctPrediction += (pred == labels).sum()

    print(f'Total : {totalPrediction}, Correct : {correctPrediction}, Acc : {(correctPrediction / totalPrediction):.2%}')

outPath = './my-mlp.pth'
torch.save(model.state_dict(), f = outPath)
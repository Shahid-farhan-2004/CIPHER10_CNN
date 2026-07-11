import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_data=datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
test_data=datasets.CIFAR10(root="./data",train=False,transform=transform,download=True)

train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=1000)

class CIPHAR10CNN(nn.Module):
    def __init__(self):
        super(CIPHAR10CNN,self).__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*8*8,256)
        self.out=nn.Linear(256,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        return self.out(x)

model=CIPHAR10CNN()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(5):
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss} in epoch {epoch+1}")

correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predictions==labels).sum().item()
    print(f"correctness is {(correct/total)}")


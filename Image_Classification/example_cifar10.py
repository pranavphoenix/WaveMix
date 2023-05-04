#install the required libraries like wavemix, torchmetrics, lion-pytorch

import torch, time
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import torch.nn as nn
from torchmetrics.classification import Accuracy
from lion_pytorch import Lion
import wavemix
from wavemix.classification import WaveMix

#use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#For small datasets with low resolution images, use WaveMix with level 1 and use strided convolutions in initial layer with stride of 1

#Model 
model = WaveMix(
    num_classes = 10,
    depth = 7,
    mult = 2,
    ff_channel = 144,
    final_dim = 144,
    dropout = 0.5,
    level=1,
    initial_conv = 'strided',
    stride = 1

)

model.to(device)
#summary
print(summary(model, (3,32,32)))    
print(torch.cuda.get_device_properties(device))

#set batch size according to GPU 
batch_size = 512

# transforms
transform_train = transforms.Compose(
        [ transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_test = transforms.Compose(
        [ transforms.ToTensor(),
     transforms.Normalize((0.4941, 0.4853, 0.4507), (0.2468, 0.2430, 0.2618))])

#Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

#metrics
top1_acc = Accuracy(task="multiclass", num_classes=10).to(device)
top5_acc = Accuracy(task="multiclass", num_classes=10, top_k=5).to(device)

#loss
criterion = nn.CrossEntropyLoss()

#Mixed Precision training
scaler = torch.cuda.amp.GradScaler()

top1 = []
top5 = []
traintime = []
testtime = []
counter = 0

#Use AdamW or lion as the first optimizer

# optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# print("Training with AdamW")

optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
print("Training with Lion")

#load saved model
# PATH = 'model.pth'
# model.load_state_dict(torch.load(PATH))

epoch = 0
while counter < 20:   #Counter sets the number of epochs of non improvement before stopping

    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
  
          inputs, labels = data[0].to(device), data[1].to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          with torch.cuda.amp.autocast():
              loss = criterion(outputs, labels)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          acc = (outputs.argmax(dim=1) == labels).float().mean()
          epoch_accuracy += acc / len(trainloader)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    correct_1=0
    correct_5=0
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels)
            correct_5 += top5_acc(outputs, labels)
            c += 1
        
    print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} - Top 5: {correct_5*100/c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    top1.append(correct_1*100/c)
    top5.append(correct_5*100/c)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    if float(correct_1*100/c) >= float(max(top1)):
        PATH = 'model.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0

#Second Optimizer
print('Training with SGD')

model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

while counter < 20: # loop over the dataset multiple times
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
  
          inputs, labels = data[0].to(device), data[1].to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          with torch.cuda.amp.autocast():
              loss = criterion(outputs, labels)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          
          acc = (outputs.argmax(dim=1) == labels).float().mean()
          epoch_accuracy += acc / len(trainloader)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    correct_1=0
    correct_5=0
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels)
            correct_5 += top5_acc(outputs, labels)
            c += 1
        
    print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} - Top 5: {correct_5*100/c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    top1.append(correct_1*100/c)
    top5.append(correct_5*100/c)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    if float(correct_1*100/c) >= float(max(top1)):
        PATH = 'model.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0
        
print('Finished Training')
print("Results")
print(f"Top 1 Accuracy: {max(top1):.2f} -Top 5 Accuracy : {max(top5):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")

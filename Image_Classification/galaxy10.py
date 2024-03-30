import imp
# from astroNN.datasets import load_galaxy10
import h5py
from tensorflow.keras import utils
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from re import split
import wavemix, pywt
# from wavemix import Level1Waveblock, DWTForward
from einops.layers.torch import Rearrange
import torch, time
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from wavemix.classification import WaveMix
import dualopt
import torch.optim as optim
from dualopt import classification, post_train
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 10
import torch.nn as nn
from torchmetrics.classification import Accuracy
from PIL import Image
# To load images and labels (will download automatically at the first time)
# First time downloading location will be ~/.astroNN/datasets/
# images, labels = load_galaxy10()

# To get the images and labels from file
with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

# To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)




train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]



class CustomImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img =  img.astype(np.uint8)
            img = Image.fromarray(img)
            
            img = self.transform(img)
            
        return img, label
    
transform_train = transforms.Compose(
        [ 
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            
     ])

trainset = CustomImageDataset(train_images, train_labels, transform=transform_train)
print(len(trainset))
testset = CustomImageDataset(test_images, test_labels, transform=transforms.ToTensor())
print(len(testset))



model = WaveMix(
    num_classes = 10,
    depth = 16,
    mult = 2,
    ff_channel = 192,
    final_dim = 192,
    dropout = 0.5,
    level = 3,
    initial_conv = 'pachify',
    patch_size = 4
)



url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/galaxy10/galaxy_95.42.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))



model.to(device)
#summary
print(summary(model, (1,3,256,256)))  

batch_size = 160




#Dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)



testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

top1_acc = Accuracy(task="multiclass", num_classes=10).to(device)

#loss
criterion = nn.CrossEntropyLoss()

#Mixed Precision training
scaler = torch.cuda.amp.GradScaler()

top1 = []
top5 = []
traintime = []
testtime = []
counter = 0

# Use AdamW or lion as the first optimizer

optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
print("Training with AdamW")

# # optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
# # print("Training with Lion")

# load saved model

PATH = 'galaxy.pth'


# model.load_state_dict(torch.load(PATH))
epoch = 0
while counter < 10:   #Counter sets the number of epochs of non improvement before stopping

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

          acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
          epoch_accuracy += acc / len(trainloader)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    
    correct_1=0
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels.argmax(dim=1))
     
            c += 1
        
    print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    top1.append(correct_1*100/c)
   
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    if float(correct_1*100/c) >= float(max(top1)):
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0

# Second Optimizer
print('Training with SGD')

model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

counter = 0
epoch = 0
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
          
          acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
          epoch_accuracy += acc / len(trainloader)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    correct_1=0
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels.argmax(dim=1))
            
            c += 1
        
    print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    top1.append(correct_1*100/c)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    if float(correct_1*100/c) >= float(max(top1)):
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0
        
print('Finished Training')
print("Results")
print(f"Top 1 Accuracy: {max(top1):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")

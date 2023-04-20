
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

import torch.nn as nn
import functools

from math import ceil
import pywt
from tqdm import tqdm 
import torch.optim as optim
import time
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms

from torchsummary import summary
import torchmetrics
from datasets import load_dataset
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

dataset_train = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='train', cache_dir = '/workspace/')
dataset_val = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
# dataset_val = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
# dataset_val = load_dataset('eugenesiow/BSD100', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
# dataset_val = load_dataset('eugenesiow/Urban100', 'bicubic_x2', split='validation', cache_dir = '/workspace/')

class SuperResolutionDataset(Dataset):
    def __init__(self, dataset, transform_img=None, transform_target=None):
        self.dataset = dataset
        self.transform_img = transform_img
        self.transform_target = transform_target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]["lr"]
        image = Image.open(img_path)

        target_path = self.dataset[idx]["hr"] 
        target = Image.open(target_path)

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_target:
            target = self.transform_target(target)


        return image, target


transform_img = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            # transforms.Normalize((0.4485, 0.4375, 0.4046), (0.2698, 0.2557, 0.2802))
     ])

transform_target = transforms.Compose(
        [transforms.Resize([512,512]),
            transforms.ToTensor(),
            # transforms.Normalize((0.4485, 0.4375, 0.4046), (0.2698, 0.2557, 0.2802))
     ])

trainset = SuperResolutionDataset(dataset_train, transform_img, transform_target)
valset = SuperResolutionDataset(dataset_val, transform_img, transform_target)


import wavemix
from wavemix import Level1Waveblock

class WaveMix(nn.Module):
    def __init__(
        self,
        *,
        depth,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.expand = nn.Sequential(
            nn.ConvTranspose2d(final_dim,int(final_dim/2), 4, stride=2, padding=1),
            nn.Conv2d(int(final_dim/2), 3, 1)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )

        
    def forward(self, img):
        x = self.conv(img)  
        
            
        for attn in self.layers:
            x = attn(x) + x

        out = self.expand(x)

        return out

model = WaveMix(
    depth = 4,
    mult = 2,
    ff_channel = 128,
    final_dim = 128,
    dropout = 0.5
)
model.to(device)
print(summary(model, (3,512,512)))

scaler = torch.cuda.amp.GradScaler()

batch_size = 16

testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size*2,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

ssim_module = SSIM(data_range=255, size_average=True, channel=3)

criterion1 =  nn.L1Loss() 
criterion2 =  nn.MSELoss() 
lamda = 0.5
scaler = torch.cuda.amp.GradScaler()
top1 = []
traintime = []
testtime = []
counter = 0
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False) #Frist optimizer
epoch = 0
while counter < 25:
    
    t0 = time.time()
    epoch_psnr = 0
    epoch_loss = 0
    
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
    
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            data_range = outputs.max().unsqueeze(0)
            with torch.cuda.amp.autocast():
                loss =  1 - ssim_module(outputs, labels) #+ criterion1(outputs, labels) + lamda*criterion2(outputs, labels) 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_PSNR = psnr(outputs, labels) 
            
            epoch_loss += loss / len(trainloader)
            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - PSNR: {epoch_PSNR:.4f}" )

    
    PSNR = 0
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            
            PSNR += psnr(outputs, labels) / len(testloader)
            
    print(f"Epoch : {epoch+1} - PSNR: {PSNR:.2f}  - Test Time: {time.time() - t1:.0f}\n")

    top1.append(PSNR)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1

    if float(PSNR) >= float(max(top1)):
        PATH = 'superresolutionDiv2k2x.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0

PATH = 'superresolutionDiv2k2x.pth'
model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #Second Optimizer

for epoch in range(1):  # loop over the dataset multiple times
    t0 = time.time()
    epoch_psnr = 0
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
              loss =  1 - ssim_module(outputs, labels) #+ criterion1(outputs, labels) + lamda*criterion2(outputs, labels) 
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          epoch_PSNR = psnr(outputs, labels)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - PSNR: {epoch_PSNR:.4f}" )
    
    t1 = time.time()
    model.eval()
    PSNR = 0   
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            PSNR += psnr(outputs, labels) / len(testloader)
     
    print(f"Epoch : {epoch+1} - PSNR: {PSNR:.2f}  - Test Time: {time.time() - t1:.0f}\n")

    top1.append(PSNR)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)

    if float(PSNR) >= float(max(top1)):
        PATH = 'superresolutionDiv2k2x.pth'
        torch.save(model.state_dict(), PATH)
        print(1)

print('Finished Training')
model.load_state_dict(torch.load(PATH))
SSIM = 0

with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            SSIM += float(ssim(outputs, labels)) / len(testloader)

print(f"Top PSNR : {max(top1):.2f} -Top SSIM : {SSIM:.4f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")

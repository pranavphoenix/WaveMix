import torch

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import wavemix
from wavemix.classification import WaveMix
import dualopt
from dualopt import classification, post_train
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True
#use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))


num_classes = 200


model = WaveMix(
    num_classes = 200,
    depth = 16,
    mult = 2,
    ff_channel = 192,
    final_dim = 192,
    dropout = 0.5,
    level = 3,
    initial_conv = 'pachify',
    patch_size = 4
)


url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/tinyimagenet/tiny_71.69.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))

model.to(device)
#summary
print(summary(model, (3,64,64)))  

PATH = 'tiny.pth' # path to save model
# torch.save(model.state_dict(), PATH)



print("ImageNet Weights Loaded")

#set batch size according to GPU 
batch_size = 512

# transforms
transform_train_1 = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
     transforms.Normalize((0.4803, 0.4481, 0.3975), (0.2764, 0.2689, 0.2816))])

transform_train_2 = transforms.Compose(
        [ 
           
        #  transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
     transforms.Normalize((0.4803, 0.4481, 0.3975), (0.2764, 0.2689, 0.2816))])

transform_test = transforms.Compose(
        [ 
            
            transforms.ToTensor(),
     transforms.Normalize((0.4825, 0.4499, 0.3984), (0.2764, 0.2691, 0.2825))])


id_dict = {}
for i, line in enumerate(open('/workspace/tiny-imagenet-200/wnids.txt', 'r')):
  id_dict[line.replace('\n', '')] = i

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/workspace/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)

        if image.mode == "L":
          image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image)
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/workspace/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('/workspace/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == "L":
          image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label




batch_size =  304



#Dataset
trainset_1 = TrainTinyImageNetDataset(id=id_dict, transform = transform_train_1)
trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

trainset_2 = TrainTinyImageNetDataset(id=id_dict, transform = transform_train_2)
trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = TestTinyImageNetDataset(id=id_dict, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

top1 = []   #top1 accuracy
top5 = []   #top5 accuracy
traintime = []
testtime = []
counter = 20  # number of epochs without any improvement in accuracy before we stop training for each optimizer



# model.load_state_dict(torch.load(PATH))

classification(model, trainloader_1, testloader, device, PATH, top1, top5, traintime, testtime,  num_classes = num_classes, set_counter = counter)
print('Finished Training')

model.load_state_dict(torch.load(PATH))

post_train(model, trainloader_2, testloader, device, PATH, top1, top5, traintime, testtime, num_classes = num_classes, set_counter = counter)
print('Finished Training')

print("Results")
print(f"Top 1 Accuracy: {max(top1):.2f} -Top 5 Accuracy : {max(top5):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")

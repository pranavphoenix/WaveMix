import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import wavemix
from wavemix.classification import WaveMix
import dualopt
from dualopt import classification, post_train

#use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

num_classes = 10

#Model 
model = WaveMix(
    num_classes = num_classes,
    depth = 16,
    mult = 2,
    ff_channel = 192,
    final_dim = 192,
    dropout = 0.5,
    level = 3,
    initial_conv = 'pachify',
    patch_size = 4

)

model.to(device)
#summary
print(summary(model, (3,192, 192)))  

PATH = 'cifar10.pth' # path to save model
torch.save(model.state_dict(), PATH)


#loading ImageNet Weights
url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/imagenet_71.49.pth'

checkpoint = torch.load(PATH)
pretrained = torch.hub.load_state_dict_from_url(url)

pretrained = {k: v for k, v in pretrained.items() if k in checkpoint and k != "pool.2.bias" and k != "pool.2.weight"}

checkpoint.update(pretrained)
model.load_state_dict(checkpoint)

print("ImageNet Weights Loaded")

#set batch size according to GPU 
batch_size = 512

# transforms
transform_train_1 = transforms.Compose(
        [ transforms.Resize([192, 192]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_train_2 = transforms.Compose(
        [ transforms.Resize([192, 192]),
         transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_test = transforms.Compose(
        [ transforms.Resize([192, 192]),
            transforms.ToTensor(),
     transforms.Normalize((0.4941, 0.4853, 0.4507), (0.2468, 0.2430, 0.2618))])

#Dataset
trainset_1 = torchvision.datasets.CIFAR10(root='/workspace/', train=True, download=True, transform=transform_train_1)
trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

trainset_2 = torchvision.datasets.CIFAR10(root='/workspace/', train=True, download=True, transform=transform_train_2)
trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = torchvision.datasets.CIFAR10(root='/workspace/', train=False, download=True, transform=transform_test)
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

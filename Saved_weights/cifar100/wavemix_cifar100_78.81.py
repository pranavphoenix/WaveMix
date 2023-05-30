import torch, wavemix
from wavemix.classification import WaveMix

model = WaveMix(
    num_classes = 100,
    depth = 10,
    mult = 2,
    ff_channel = 256,
    final_dim = 256,
    dropout = 0.5,
    level=1,
    initial_conv = 'strided',
    stride = 1

)

url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/cifar100/cifar100_78.81.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))
import torch, wavemix
from wavemix.classification import WaveMix

model = WaveMix(
    num_classes = 1000,
    depth = 16,
    mult = 2,
    ff_channel = 256,
    final_dim = 256,
    dropout = 0.5,
    level = 4,
    initial_conv = 'strided',
    stride = 2,

)

url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/imagenet4level_70.42.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))
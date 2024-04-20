import torch, wavemix
from wavemix.classification import WaveMix

model = WaveMix(
    num_classes = 1000,
    depth = 16,
    mult = 2,
    ff_channel = 192,
    final_dim = 192,
    dropout = 0.5,
    level = 1,
    initial_conv = 'pachify',
    patch_size = 4
)

url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/wavemixlite_71.18.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))

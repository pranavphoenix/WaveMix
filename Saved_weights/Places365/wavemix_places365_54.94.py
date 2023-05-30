import torch, wavemix
from wavemix.classification import WaveMix


model = WaveMix(
    num_classes = 365,
    depth = 12,
    mult = 2,
    ff_channel = 256,
    final_dim = 256,
    dropout = 0.5,
    level = 2,
    initial_conv = 'pachify',
    patch_size = 8

)

url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/Places365/places365_54.94.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))
import torch, wavemix
from wavemix.classification import WaveMix

model = WaveMix(
    num_classes = 10,
    depth = 16,
    mult = 2,
    ff_channel = 192,
    final_dim = 192,
    dropout = 0.5,
    level = 1,
    initial_conv = 'pachify',
    patch_size = 4
)

url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/galaxy10/galaxy_90.23.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))

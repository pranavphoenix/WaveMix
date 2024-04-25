import torch, wavemix
from wavemix.classification import WaveMix

model = WaveMix(
    num_classes = 1000,
    depth = 16,
    mult = 2,
    ff_channel = 192,
    final_dim = 192,
    dropout = 0.5,
    level = 3,
    initial_conv = 'pachify',
    patch_size = 4
)

model.pool = nn.Sequential(
            nn.ConvTranspose2d(192 , 128, 4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.Conv2d(64, 20, 1) 
)

url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/wavemix_192_16_75.06.pth'

model.load_state_dict(torch.hub.load_state_dict_from_url(url))

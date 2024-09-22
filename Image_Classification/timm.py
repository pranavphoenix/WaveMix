"""
WaveMix models for image classification.
Some implementations are modified from:
timm (https://github.com/rwightman/pytorch-image-models),
MambaOut (https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py)

Also add in the models/init file in timm
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from wavemix import Level1Waveblock, Level2Waveblock, Level3Waveblock, Level4Waveblock
from einops.layers.torch import Rearrange

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'wavemix': _cfg(
        url='https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/wavemix_192_16_75.06.pth'),
    'wavemix_lite': _cfg(
        url='https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/wavemixlite_71.18.pth'),
   
}

class WaveMix(nn.Module):
    def __init__(
        self,
        *,
        num_classes=1000,
        depth = 16,
        mult = 2,
        ff_channel = 192,
        final_dim = 192,
        dropout = 0.5,
        level = 3,
        initial_conv = 'pachify', # or 'strided'
        patch_size = 4,
        stride = 2,

    ):
        super().__init__()

        self.num_classes = num_classes
        
        self.layers = nn.ModuleList([])
        for _ in range(depth): 
                if level == 4:
                    self.layers.append(Level4Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 3:
                    self.layers.append(Level3Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 2:
                    self.layers.append(Level2Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                else:
                    self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(final_dim, num_classes)
        )

        if initial_conv == 'strided':
            self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, stride, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, stride, 1)
        )
        else:
            self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/4),3, 1, 1),
            nn.Conv2d(int(final_dim/4), int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, patch_size, patch_size),
            nn.GELU(),
            nn.BatchNorm2d(final_dim)
            )
        
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        x = self.conv(x)   
        for attn in self.layers:
            x = attn(x) + x
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    




###############################################################################
# a series of WaveMix models
@register_model
def wavemix(pretrained=False, **kwargs):
    model =  WaveMix(
        num_classes = 1000,
        depth = 16,
        mult = 2,
        ff_channel = 192,
        final_dim = 192,
        dropout = 0.5,
        level = 3,
        initial_conv = 'pachify',
        patch_size = 4)
    model.default_cfg = default_cfgs['wavemix']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model



@register_model
def wavemix_lite(pretrained=False, **kwargs):
    model = WaveMix(
    num_classes = 1000,
    depth = 16,
    mult = 2,
    ff_channel = 192,
    final_dim = 192,
    dropout = 0.5,
    level = 1,
    initial_conv = 'pachify',
    patch_size = 4,
        **kwargs)
    model.default_cfg = default_cfgs['wavemix_lite']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model



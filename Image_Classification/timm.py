#Code to use in Timm Library
#Replace the code in convmixer.py in pytorch-image-models/timm/models/ with this code and run

""" ConvMixer

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import pywt
import functools
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import pywt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from .helpers import build_model_with_cfg, checkpoint_seq
from .layers import SelectAdaptivePool2d


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        'first_conv': 'stem.0',
        **kwargs
    }


device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wavemix
from wavemix import Level4Waveblock, Level3Waveblock, Level2Waveblock, Level1Waveblock
import torch.nn as nn
from einops.layers.torch import Rearrange


class ConvMixer(nn.Module):
    def __init__(
            self, depth, mult = 2, ff_channel = 16, final_dim = 16, dropout = 0., level =3, patch_size = 4, in_chans=3, num_classes=1000, global_pool='avg',
            act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = final_dim
        self.grad_checkpointing = False

        self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/4),3, 1, 1),
            nn.Conv2d(int(final_dim/4), int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, patch_size, patch_size),
            nn.GELU(),
            nn.BatchNorm2d(final_dim)
            )
        
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

        
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(final_dim, num_classes)
        )
       

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(conv=r'^conv', layers=r'^layers\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

          
    def forward_features(self, x):
        x = self.conv(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.layers, x)
        else:
            for attn in self.layers:
                x = attn(x) + x
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.pool(x)
        return x 

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_convmixer(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ConvMixer, variant, pretrained, **kwargs)


@register_model
def convmixer(pretrained=False, **kwargs):
    model_args = dict(depth = 16, mult = 2, ff_channel = 192, final_dim = 192, dropout = 0.5, level = 3, patch_size = 4, **kwargs)
    return _create_convmixer('convmixer', pretrained, **model_args)


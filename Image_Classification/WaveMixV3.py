import torch, math
import torch.nn as nn
from lion_pytorch import Lion
import wavemix
from wavemix import DWTForward
# from wavemix.classification import WaveMix
import numpy as np
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange


     
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

xf1 = DWTForward(J=1, mode='zero', wave='db1').to(device)    

class Waveblock(nn.Module):
    def __init__(
        self,
        *,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.5,
    ):
        super().__init__()
        
        self.feedforward = nn.Sequential(
                nn.Conv2d(final_dim, final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.PixelShuffle(2),
                nn.Conv2d(final_dim, final_dim,3, 1, 1),
                nn.BatchNorm2d(final_dim)
            
            )
 
        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.reduction(x)
        
        Y1, Yh = xf1(x)

        x = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        
        x = torch.cat((Y1,x), dim = 1)

        x = self.feedforward(x)

        return x
    
class WaveMix(nn.Module):
    def __init__(
        self,
        *,
        num_classes=1000,
        depth = 16,
        mult = 4,
        ff_channel = 192,
        final_dim = 192,
        dropout = 0.5,
        level = 3,
        initial_conv = 'patch', # or 'strided'
        patch_size = 4,
        stride = 1,

    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth): 
            self.layers.append(Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.pool = nn.Sequential(
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
        
    def forward(self, img):
        x = self.conv(img)   
            
        for attn in self.layers:
            x = attn(x) + x

        out = self.pool(x)

        return out

model = WaveMix(
    num_classes= 1000, 
    depth= 16,
    mult= 4,
    ff_channel= 144,
    final_dim= 144,
    dropout= 0.5,
    level=1,
    patch_size=4,
)
model.to(device)

print(summary(model, (1, 3, 224, 224)))


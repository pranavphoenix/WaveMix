# -*- coding: utf-8 -*-

import torch, wavemix
from wavmix import Level1Waveblock


class WaveMix(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        depth,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(final_dim, num_classes)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )

      

    def forward(self, img):
        x = self.conv(img)   
            
        for attn in self.layers:
            x = attn(x) + x

        out = self.pool(x)

        return out

model = WaveMix(
    num_classes = 1000,
    depth = 7,
    mult = 2,
    ff_channel = 256,
    final_dim = 256,
    dropout = 0.5
)


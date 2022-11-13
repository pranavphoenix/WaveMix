# With 2 skip connections from input convolution layers to output deconvolution layers

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

import sys, os, time, pickle
import numpy as np
import math
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

import torch.optim as optim

from torchsummary import summary

from math import ceil
import pywt

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
from einops import reduce


dataset = torchvision.datasets.Cityscapes('/workspace/', split='train', mode='fine',
                      target_type='semantic')
dataset[0][0].size

fig,ax=plt.subplots(ncols=2,figsize=(12,8))
ax[0].imshow(dataset[0][0])
ax[1].imshow(dataset[0][1],cmap='gray')

ignore_index=255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']


class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)
class_map

colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(n_classes), colors))

def encode_segmap(mask):
    #remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

# !pip uninstall opencv-python-headless==4.5.5.62



import albumentations as A
from albumentations.pytorch import ToTensorV2

transform=A.Compose(
[
    A.Resize(1024, 2048),
    # A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]
)

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes

class MyClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed=transform(image=np.array(image), mask=np.array(target))            
        return transformed['image'],transformed['mask']

dataset=MyClass('/workspace/', split='val', mode='fine',
                     target_type='semantic',transforms=transform)
img,seg= dataset[20]
print(img.shape,seg.shape)

fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,8))
ax[0].imshow(img.permute(1, 2, 0))
ax[1].imshow(seg,cmap='gray')

print(torch.unique(seg))
print(len(torch.unique(seg)))

res=encode_segmap(seg.clone())
print(res.shape)
print(torch.unique(res))
print(len(torch.unique(res)))

res1=decode_segmap(res.clone())

fig,ax=plt.subplots(ncols=2,figsize=(12,10))  
ax[0].imshow(res,cmap='gray')
ax[1].imshow(res1)

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import pywt
import torch.nn as nn
import functools
import segmentation_models_pytorch as smp

# !pip uninstall torchmetrics

import torchmetrics
###################MODEL################################################################

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L-2, 0) if d == 2 else (0, L-2)
            y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y

def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image
    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
            else:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L-1, 0) if d == 2 else (0, L-1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N//2
        if d == 2:
            lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
            lohi = lohi[:,:,:N2]
        else:
            lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p//2, 0) if d == 2 else (0, p//2)
            # Calculate the high and lowpass
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p//2, (p+1)//2) if d == 2 else (p//2, (p+1)//2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi



class AFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None


def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=device):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_col = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row


def prep_filt_afb1d(h0, h1, device=device):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0 (array-like): low pass column filter bank
        h1 (array-like): high pass column filter bank
        device: which device to put the tensors on to
    Returns:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1

class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh

from numpy.lib.function_base import hamming
xf1 = DWTForward(J=1, mode='zero', wave='db1').cuda()
xf2 = DWTForward(J=2, mode='zero', wave='db1').cuda() 
xf3 = DWTForward(J=3, mode='zero', wave='db1').cuda()
xf4 = DWTForward(J=4, mode='zero', wave='db1').cuda()  
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
        
      
        self.feedforward1 = nn.Sequential(
                nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, ff_channel, 1),
                nn.ConvTranspose2d(ff_channel, final_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(final_dim)         
            )

        self.feedforward2 = nn.Sequential(
                nn.Conv2d(final_dim + int(final_dim/2), final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, ff_channel, 1),
                nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
                nn.BatchNorm2d(int(final_dim/2))            
            )

        self.feedforward3 = nn.Sequential(
                nn.Conv2d(final_dim+ int(final_dim/2), final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, ff_channel, 1),
                nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
                nn.BatchNorm2d(int(final_dim/2))          
            )

        self.feedforward4 = nn.Sequential(
                nn.Conv2d(final_dim, final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, ff_channel, 1),
                nn.ConvTranspose2d(ff_channel, int(final_dim/2), 4, stride=2, padding=1),
                nn.BatchNorm2d(int(final_dim/2))          
            )    

        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
        
        
    def forward(self, x):
        b, c, h, w = x.shape
  
        x = self.reduction(x)
        
        Y1, Yh = xf1(x)
        Y2, Yh = xf2(x)
        Y3, Yh = xf3(x)
        Y4, Yh = xf4(x)
        
        x1 = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        x2 = torch.reshape(Yh[1], (b, int(c*3/4), int(h/4), int(w/4)))
        x3 = torch.reshape(Yh[2], (b, int(c*3/4), int(h/8), int(w/8)))
        x4 = torch.reshape(Yh[3], (b, int(c*3/4), int(h/16), int(w/16)))
        
        x1 = torch.cat((Y1,x1), dim = 1)
        x2 = torch.cat((Y2,x2), dim = 1)
        x3 = torch.cat((Y3,x3), dim = 1)
        x4 = torch.cat((Y4,x4), dim = 1)
        
        
        x4 = self.feedforward4(x4)
        
        x3 = torch.cat((x3,x4), dim = 1)
        
        x3 = self.feedforward3(x3)
        
        x2 = torch.cat((x2,x3), dim = 1)

        x2 = self.feedforward2(x2)

        x1 = torch.cat((x1,x2), dim = 1)
        x = self.feedforward1(x1)
    
        return x

class WaveMix(nn.Module):
    def __init__(
            self,
            *,
            num_classes,
            depth,
            mult =2,
            ff_channel = 16,
            final_dim =16,
            dropout = 0.,
        ):

            super().__init__()

            self.layers = nn.ModuleList([]) 
            for _ in range(depth): 
                self.layers.append(Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))                    
            self.deconv1 = nn.ConvTranspose2d(final_dim,int(final_dim/2), 4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(int(final_dim/2)+int(final_dim/4), int(final_dim/4), 4, stride=2, padding=1)
              #  nn.Conv2d(int(final_dim/4), num_classes, 1)
            

            self.conv1 = nn.Conv2d(3, int(final_dim/2), 3, 2, 1)
            self.conv2 = nn.Conv2d(int(final_dim/2),final_dim, 3, 2, 1)
            
            self.final = nn.Conv2d(int(final_dim/4)+3, num_classes, 1) 
            self.compress = nn.Conv2d(int(final_dim/2), int(final_dim/4), 1, 1) 

    def forward(self, img):
        x1 = self.conv1(img)
        x = self.conv2(x1)
        x1 = self.compress(x1)

        for attn in self.layers:
            x = attn(x) + x

        out = self.deconv1(x)
        out = torch.cat((out, x1), dim=1)
        out = self.deconv2(out)
        out = torch.cat((out, img), dim=1)
        out = self.final(out)
        return out
        
           

model = WaveMix(
    num_classes = 20,
    depth = 16,
    mult = 2,
    ff_channel = 256,
    final_dim = 256,
    dropout = 0.5
)

model.to(device)
print(summary(model, (3,1024, 2048)))    
print(torch.cuda.get_device_properties(device))

from torch.autograd import Variable
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        loss = -((1-pt)**self.gamma) * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

#criterion= smp.losses.DiceLoss(mode='multiclass')
metrics = torchmetrics.IoU(num_classes=n_classes)

trainset = MyClass('/workspace/', split='train', mode='fine',
                     target_type='semantic',transforms=transform)
testset= MyClass('/workspace/', split='val', mode='fine',
                     target_type='semantic',transforms=transform)

batch_size = 5


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = outputs.argmax(dim=1)

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
# PATH = '/workspace/CityscapesclasswaveMix256layer16_512by1028skip2.pth'
# model.load_state_dict(torch.load(PATH))

print(len(trainset))
print(len(testset))
criterion = FocalLoss2d()
scaler = torch.cuda.amp.GradScaler()
from tqdm import tqdm
miou = []
epoch_losses = []
test_losses = []
traintime = []
testtime = []
counter = 0
epoch = 0
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
while counter < 25:  # loop over the dataset multiple times
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0

    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:

            inputs, labels = data[0].to(device), data[1].to(device)
            # labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            segment=encode_segmap(labels)

            with torch.cuda.amp.autocast():
                loss = criterion(outputs, segment.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)
        
           
            running_loss += loss.item()

            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    epoch_losses.append(epoch_loss)
    test_loss = 0
    total = 0

    mIoU = 0
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # labels = labels.squeeze(1).long()
            outputs = model(images)
            segment=encode_segmap(labels)
#         outputs = net(images)

            # _, predicted = torch.max(outputs.data, 1)
            with torch.cuda.amp.autocast():
              test_loss += criterion(outputs, segment.long())
              # mIoU = metrics(outputs, labels)
              # mIoU +=  metrics(outputs,segment)
            mIoU += iou_pytorch(outputs, segment).mean()
          
    
    mIoU = mIoU/len(testloader)
    test = test_loss/len(testloader)
    test_losses.append(test)
    mIoU = mIoU.cpu().detach()
    miou.append(mIoU)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1

    print(f"Epoch : {epoch+1} - MIOU: {mIoU:.4f} -Test Time: {time.time() - t1:.0f} \n")
    if mIoU >= max(miou):
        PATH = '/workspace/CityscapesclasswaveMix256layer16_512by1028skip2.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0
    
print('Finished Training')

model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(100):  # loop over the dataset multiple times
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0

    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:

    
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            segment=encode_segmap(labels)

            with torch.cuda.amp.autocast():
                loss = criterion(outputs, segment.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)
        
            running_loss += loss.item()

            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )
    epoch_losses.append(epoch_loss)
    test_loss = 0
    total = 0

    mIoU = 0
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
           
            outputs = model(images)
            segment=encode_segmap(labels)
#         
            with torch.cuda.amp.autocast():
              test_loss += criterion(outputs, segment.long())
              
            mIoU += iou_pytorch(outputs, segment).mean()
          
    
    mIoU = mIoU/len(testloader)
    test = test_loss/len(testloader)
    test_losses.append(test)
    mIoU = mIoU.cpu().detach()
    miou.append(mIoU)

    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)

    print(f"Epoch : {epoch+1} - MIOU: {mIoU:.4f} -Test Time: {time.time() - t1:.0f} \n")
    if mIoU >= max(miou):
        PATH = '/workspace/CityscapesclasswaveMix256layer16_512by1028skip2.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
    
print('Finished Training')
print(f"Top mIoU : {max(miou):.4f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")


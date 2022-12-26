# import pyshearlab
import util
import matplotlib.pyplot as plt #debug

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import geometry
from einops import rearrange
from pdb import set_trace as pdb #debug
import lpips
import kornia
from shortcut_macros import *

interp = lambda x,y:F.interpolate(x,y,mode="bilinear",align_corners=True)

highsl=128

if 'loss_fn_alex' not in globals():
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
def tmp(model_out,gt,model_in):
    print(rgb(model_out,gt,model_in))
    print(percept(model_out,gt,model_in))
    print(multiscale_rgb(model_out,gt,model_in))
    print(depth(model_out,gt,model_in))
    print(depth_grad(model_out,gt,model_in))
    print(peaky_surf(model_out,gt,model_in))

def depth_grad(model_out,gt,model_in):
    estgrad = kornia.filters.spatial_gradient(ch_fst(model_out["depth"]),normalized=False).abs()
    return (estgrad).square().mean()

def peaky_surf(model_out,gt,model_in):
    return (1-model_out["weights"].max(dim=-1)[0]).mean()

def depth(model_out,gt,model_in):
    depth=model_out["depth"]
    depthgt=model_out["gtdepth"].to(depth)
    return ((1/(depth.clip(1.5,300)+1e-5)-1/(depthgt.clip(1.5,300)+1e-5))).abs().mean()
    #return (((depth.clip(1.5,300)+1e-5)-(depthgt.clip(1.5,300)+1e-5))).abs().mean()*1e-2

def percept(model_out,gt,model_in):
    gtrgb,rgb=ch_fst(model_out["gtrgb"]), ch_fst(model_out["rgb"])
    return loss_fn_alex(gtrgb,rgb).mean()
    
def multiscale_rgb(model_out,gt,model_in):
    pred,gt=ch_fst(model_out["gtrgb"]), ch_fst(model_out["rgb"])
    loss=0
    for _ in range(3):
        pred = F.interpolate(pred,scale_factor=1/2)
        gt = F.interpolate(gt,scale_factor=1/2)
        loss += (pred-gt).square().mean()
    return loss

def rgb(model_out,gt,model_in):
    return (model_out["rgb"]-model_out["gtrgb"]).square().mean()

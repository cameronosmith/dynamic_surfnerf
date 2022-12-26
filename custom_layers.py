import math

import torch.nn.functional as F
import numpy as np
import geometry
import torchvision
import util
from collections import OrderedDict
from torch.nn.init import _calculate_correct_fan

from pdb import set_trace as pdb

import torch
from torch import nn

from einops import repeat,rearrange

from attn_modules import CrossAttn_

ch_fst = lambda x:rearrange(x,"... (x y) c -> ... c x y",x=int(x.size(-2)**(.5)))
ch_sec = lambda x:rearrange(x,"... c x y -> ... (x y) c")
interp = lambda x,y:F.interpolate(x,y,mode="bilinear",align_corners=True)
normpool = lambda x,y,dim=-1: torch.where(x.norm(dim=dim,keepdim=True)>y.norm(dim=dim,keepdim=True),x,y)

class ResMLP(nn.Module):
    def __init__(self, ch_in, ch_mod, out_ch, num_res_block=1 ):
        super().__init__()

        self.res_blocks = nn.ModuleList([
          nn.Sequential(nn.Linear(ch_mod,ch_mod),nn.ReLU(),
                        nn.LayerNorm([ch_mod], elementwise_affine=True),
                        nn.Linear(ch_mod,ch_mod),nn.ReLU())
            for _ in range(num_res_block)
        ])  

        self.proj_in = nn.Linear(ch_in,ch_mod)
        self.out = nn.Linear(ch_mod,out_ch)

    def forward(self,x):

        x = self.proj_in(x)

        for i,block in enumerate(self.res_blocks):

            x_in = x

            x = block(x)

            if i!=len(self.res_blocks)-1: x = x + x_in

        return self.out(x)

# Not really a true UNet - just downconv into transformer into upconv and pool on the original vs output
class TransformerUNet(nn.Module):
    def __init__(self, num_down=1,num_up=1,ch=256,lowx=16,lowy=16,num_self_attns=4):
        super().__init__()

        self.self_attns = nn.ModuleList([ CrossAttn_(ch,heads=4,dim_head=ch//4) for _ in range(num_self_attns) ])
        self.lowx=lowx
        self.lowy=lowy
        self.pos_embx=nn.Parameter(torch.rand(lowx,ch//2))
        self.pos_emby=nn.Parameter(torch.rand(lowy,ch//2))
        downconvs=[]
        upconvs=[]
        for i in range(num_down):
            downconvs.append(nn.Conv2d(ch,ch,3,2,1))
        for i in range(num_up):
            upconvs.append(nn.Conv2d(ch*2,ch,3,1,1))
        self.downconvs = nn.Sequential(*downconvs)#.cuda()
        self.upconvs = nn.Sequential(*upconvs)#.cuda()
        self.img_lin=nn.Linear(ch,ch)#.cuda()

    def forward(self,img):
        downconv_feats=[img]
        for conv in self.downconvs:
            downconv_feats.append( conv(downconv_feats[-1]).relu() )
        glob_pos = torch.cat(( repeat(self.pos_embx,"y c -> b c x y",b=img.size(0),x=self.lowy),
                               repeat(self.pos_emby,"x c -> b c x y",b=img.size(0),y=self.lowx),
                                ),1)
        tokens=self.img_lin(ch_sec(glob_pos+downconv_feats[-1]))
        for attn in self.self_attns: 
            tokens=attn(tokens,tokens)
        tokens=tokens.permute(0,2,1).unflatten(-1,(self.lowy,self.lowx))
        for upconv,downconv_feat in zip(self.upconvs,reversed(downconv_feats[:-1])):
            tokens = F.interpolate(tokens,scale_factor=2,mode="nearest")
            tokens=upconv(torch.cat((tokens,downconv_feat),1)).relu()
        return tokens

def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

def sal_init(m):
    if type(m) == BatchLinear or nn.Linear:
        if hasattr(m, 'weight'):
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out'))

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)

def sal_init_last_layer(m):
    if hasattr(m, 'weight'):
        val = np.sqrt(np.pi) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_in'))
        with torch.no_grad():
            m.weight.fill_(val)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')



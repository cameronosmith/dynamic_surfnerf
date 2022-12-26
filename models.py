import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from einops import repeat,rearrange
from pdb import set_trace as pdb #debugging

import util, custom_layers, conv_modules, geometry

import numpy as np

import torch
F=torch.nn.functional
from torch import nn
import features

from shortcut_macros import *

class DynamicSurfNerf(nn.Module):
    def __init__(self,backbone="fpn",num_tok=2): 
        super().__init__()

        latent_dim=512

        if backbone is "midas":
            self.backbone = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            self.backbone.scratch.output_conv = nn.Sequential()
            self.backbone_linear=nn.Linear(256,latent_dim)
        else:
            self.backbone=conv_modules.FeaturePyramidEncoder()
            self.backbone_linear=nn.Linear(latent_dim,latent_dim)

        key_dim = 13
        self.depth_key_dim=key_dim
        self.num_tok = num_tok
        self.pos_emb_tok = nn.Parameter(torch.rand(self.num_tok,latent_dim),requires_grad=True)

        num_freq_bands=(key_dim-1)//2
        self.pos_encoder_z = features.PositionalEncoding(1,num_freq_bands)
        self.depth_scaler = nn.Parameter(torch.rand(key_dim),requires_grad=True) 

        self.phi = custom_layers.ResMLP(512,512,key_dim+3,3).cuda()
        self.phi.apply(custom_layers.init_weights_normal)

    def pix_to_surf(self,ctxt_rgb):

        # First backbone features. For now just using first image, later need all images for motion
        backbone_feats = self.backbone_linear(ch_sec(interp(self.backbone(ctxt_rgb),ctxt_rgb.shape[-2:])))
        pix_tok = self.phi(self.pos_emb_tok[None,None]+backbone_feats[:,:,None])

        # Set first token colors to rgb and apply activation to second token colors
        back_colors = pix_tok[...,1:,-3:].tanh()
        front_rgb = ctxt_rgb.permute(0,2,3,1).flatten(1,2)[...,None,:]
        pix_tok = rearrange(torch.cat((torch.cat((pix_tok[...,:1,:-3],front_rgb),-1),
                                       torch.cat((pix_tok[...,1:,:-3],back_colors),-1),
                                ),2),"b (x y) t c -> b (t c) x y",x=ctxt_rgb.size(-2))
        return pix_tok

    def surfnerf_render(self, img_samp, cam_depths, samp_dists, white_back=True):

        # Unpack img_feats into the amortized k,v layers
        img_samp = rearrange(img_samp,"ctx xy s (t c) -> ctx xy s t c",c=self.depth_key_dim+3)

        depth_enc = self.depth_scaler * self.pos_encoder_z(cam_depths.unsqueeze(-1))
        depth_key,rgb = img_samp[...,:-3], img_samp[...,-3:]

        sigma = (depth_key * depth_enc).sum(dim=-1,keepdim=True)

        token_weights = sigma.softmax(dim=-2)
        sigma,rgb = [(x * token_weights).sum(dim=-2) for x in (sigma,rgb)]
        
        rgb,estdepth,weights = util.volume_integral(samp_dists,sigma,rgb)

        if white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1.0 - accum)

        max_surface_vis = (token_weights.squeeze(-1)*weights).sum(2).max(dim=-1)[1]
        return rgb,estdepth,max_surface_vis,weights

    def forward(self,input,full_imgs=False, pix_tok=None, render_rays=None,crop_res=100,use_crops=True):

        (b, n_ctxt), n_qry = input["context"]["uv"].shape[:2],input["query"]["uv"].shape[1:3][0]
        if render_rays is None: 
            if use_crops: render_rays = util.get_crop_rays(*input["context"]["rgb"].shape[-2:],crop_res)
            else: render_rays = util.get_random_rays(*input["context"]["rgb"].shape[-2:],crop_res**2)

        # Encode images into multi-token image-aligned features
        if pix_tok is None:
            pix_tok = self.pix_to_surf(input["context"]["rgb"].flatten(0,1))

        # Make 3d samples - to be factored away
        samp_dists = repeat( torch.logspace(np.log10(3),np.log10(400),192).cuda(), "d -> b xy d",b=n_qry*b,xy=len(render_rays) )
        samp_world = util.get_samp_3d(input["query"]["cam2world"],input["query"]["uv"].flatten(2,3)[:,:,render_rays],
                                      input["query"]["intrinsics"].float(),samp_dists.permute(0,2,1))
        img_samp, cam_depths = util.pixel_aligned_features(
            repeat(samp_world,"(b qry) pix d c -> (b ctx qry) (pix d) c",b=b,ctx=n_ctxt),
            repeat(input["context"]["cam2world"],"b ctx x y -> (b ctx qry) x y",qry=n_qry),
            repeat(input["context"]["intrinsics"],"b ctx x y -> (b ctx qry) x y",qry=n_qry),
            repeat(pix_tok,"(b ctx) c x y -> (b ctx qry) c x y",qry=n_qry,b=b),
        )
        img_samp = rearrange(img_samp,"(b ctx qry) c (pix d) -> (b qry) pix d (ctx c)",b=b,ctx=n_ctxt,pix=samp_world.size(1))
        cam_depths = repeat(cam_depths,"(b ctx qry) (pix d) -> (b qry) pix d (ctx tok)",b=b,ctx=n_ctxt,pix=samp_world.size(1),tok=self.num_tok) 

        # Render image
        rgb,estdepth,token_weights,weights = self.surfnerf_render(img_samp,cam_depths,samp_dists)

        return {
            "depth":estdepth,
            "gtdepth":input["query"]["depth"].flatten(0,1)[:,None].flatten(-2,-1)[...,render_rays].permute(0,2,1),
            "weights":weights.squeeze(-1),
            "rgb":rgb,
            "gtrgb":input["query"]["rgb"].flatten(0,1).flatten(-2,-1)[...,render_rays].permute(0,2,1),
            "surface_token_idx":token_weights[...,None],
            "pix_tok":pix_tok,
        }

import matplotlib.colors as colors
import torch.nn.functional as F
import copy
import geometry
import time
import os, struct, math
import numpy as np
import torch
from glob import glob
import collections
import cv2
import pytorch3d
from shortcut_macros import *

from einops import repeat,rearrange
import kornia

from pdb import set_trace as pdb

from typing import Callable, List, Optional, Tuple, Generator, Dict
#from cc_torch import connected_components_labeling
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops import interpolate_face_attributes

def render_cam_traj(model,input,pix_tok=None,num_bundles=10):
    input = copy.deepcopy(input)
    input["query"]["uv"] = input["query"]["uv"][:,:1]
    input["query"]["intrinsics"] = input["query"]["intrinsics"][:,:1]
    c2w = input["query"]["cam2world"][:,:1]
    print(c2w[0,0,:3,-1])
    org_pos=c2w[0,0,:3,-1]
    tmp=torch.eye(4).cuda()
    n=20
    circ_scale = .1
    thetas=np.linspace(0,4*np.pi,n)
    for i in range(n):
        theta=thetas[i]
        x=np.cos(theta)*2 * circ_scale
        y=np.sin(theta) * circ_scale
        tmp[:3,-1] = torch.tensor([x,y,i/10]).cuda().float()
        print(tmp[:3,-1],x,y)
        input["query"]["cam2world"] = c2w@tmp[None,None]
        out = render_full_img(model,input,pix_tok)
        plt.imsave("/nobackup/users/camsmith/img/nvrots/%02d.png"%i,(out["rgb"][0].clip(-1,1)*.5+.5).permute(1,2,0).cpu().numpy())
    #imsave(out["rgb"][0].clip(-1,1))
    #imsave(out["depth"][0,0],1)

def render_full_img(model,input,pix_tok=None,num_bundles=10):
    res=input["query"]["rgb"].shape[-2:]
    rays_all = torch.arange(torch.tensor(res).prod())
    render_outs = []
    # Render rays in chunks
    for i,rays in enumerate(rays_all.chunk(num_bundles)):
        model_out = model(input,False,pix_tok,rays)
        pix_tok = model_out["pix_tok"]
        render_outs.append(model_out)
    out = {"pix_tok":pix_tok,}
    for k in filter(lambda x: x not in out.keys(),render_outs[0].keys()):
        out[k] = torch.cat([render_out[k] for render_out in render_outs],-2).permute(0,2,1).unflatten(-1,res)
    return out

def get_random_rays(imsly,imslx,num_render_rays):
    return torch.randperm(imslx*imsly)[:num_render_rays]
def get_crop_rays(imsly,imslx,crop_res):
    start_x,start_y = np.random.randint(0,imslx-crop_res),np.random.randint(0,imsly-crop_res)
    return torch.arange(imslx*imsly).view(imsly,imslx)[start_y:start_y+crop_res,start_x:start_x+crop_res].flatten()

def get_samp_3d(cam2world,uv,K,samp_dists):
    ray_dirs = geometry.get_ray_directions(uv.flatten(0,1), cam2world=cam2world.flatten(0,1), intrinsics=K.flatten(0,1)
                                             ).permute(0,2,1)
    samp_3d = ( repeat(geometry.get_ray_origin(cam2world.flatten(0,1)),"b c -> b s xy c",s=1,xy=uv.size(2)) +
                repeat(ray_dirs,"b c xy -> b s xy c",s=1,xy=uv.size(-2)) * samp_dists[...,None]).permute(0,2,1,3)
        
    return samp_3d

def volume_integral(
    z_vals: torch.tensor,
    sigmas: torch.tensor,
    radiances: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:

    # Compute the deltas in depth between the points.
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1], 
        torch.Tensor([1e10]).to(z_vals.device).expand(z_vals[...,:1].shape)
        ], -1) 

    # Compute the alpha values from the densities and the dists.
    # Tip: use torch.einsum for a convenient way of multiplying the correct 
    # dimensions of the sigmas and the dists.
    #alpha = 1.- torch.exp(-torch.einsum('brzs, z -> brzs', sigmas, dists))
    #alpha = 1.- torch.exp(-torch.einsum('brzs, brz -> brzs', sigmas, dists))
    alpha = 1.- torch.exp(-sigmas.relu()*dists.unsqueeze(-1)*1e-1)

    # Compute the Ts from the alpha values. Use torch.cumprod.
    Ts = torch.cumprod(1.-alpha + 1e-10, -2)

    # Compute the weights from the Ts and the alphas.
    weights = alpha * Ts
    
    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum('brzs, brzs -> brs', weights.expand(-1,-1,-1,3), radiances)

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    #depth_map = torch.einsum('brzs, z -> brs', weights, z_vals)
    depth_map = torch.einsum('brzs, brz -> brs', weights, z_vals)

    return rgb, depth_map, weights

# ignore -- custom google dataset cam2world to level with groundplane transformation
def google_obj_cam2ground(cam2world):
    pos=cam2world[:,:3,-1]
    level_transf=torch.stack([torch.from_numpy(geometry.look_at(torch.tensor([0,pos[0,:2].norm(),z]),0)) 
                                                for x,y,z in pos.cpu()]).cuda().float().inverse()
    trans=torch.eye(4)[None].cuda()
    #trans[:,-2,-1]=-3.6
    return trans@level_transf

def get_rotating_cams(cam2world,num_steps):

    cam2worlds=[]
    for theta_i,theta2 in enumerate(np.linspace(0,1,num_steps)):

        cam2world=cam2world.clone()

        pos=cam2world[:,0,:3,-1]
        r=pos[0,:2].norm()
        new_loc=torch.tensor([r*np.cos(2*np.pi*theta2),r*np.sin(2*np.pi*theta2), float(cam2world[0,0,2,-1])]).float()

        cam2world2=torch.from_numpy(geometry.look_at(new_loc,0,True))[None,None].cuda().float()

        cam2worlds.append(cam2world2)

    return torch.stack(cam2worlds)


# Projects world coords to camera image plane
def project(K,crds):
    crds = torch.cat((crds,torch.ones_like(crds[...,:1])),-1).permute(0,2,1)
    pix_proj = K[:,:3,:3]@crds[:,:3]
    pix_proj = ( pix_proj[:,:2]/(pix_proj[:,[2]]+1e-3) )
    return pix_proj.permute(0,2,1)

def img_proj(K,samp_3d,img_feats):
    img_crds = project(K,samp_3d)
    return grid_samp(img_feats,img_crds[:,None]*2-1).squeeze(2)#.permute(0,2,3,1)

def pixel_aligned_features(samp_world,ctxt_cam2world,intrinsics,img_feats):
    samp_cam = (ctxt_cam2world.inverse() @ hom(samp_world).permute(0,2,1))[:,:3].permute(0,2,1)
    img_samp = img_proj(intrinsics,samp_cam,img_feats)
    return img_samp,samp_cam.norm(dim=-1)

def get_wc(uv,K,cam2world,depth):
    uvh=torch.cat((uv,torch.ones_like(uv[...,:1])),-1)
    camcrds=K[:,:3,:3].inverse()@(uvh*depth).permute(0,2,1)
    camcrdsh = torch.cat((camcrds, torch.ones_like(camcrds[:,:1])),1).permute(0,2,1) 
    wc = (cam2world@camcrdsh.permute(0,2,1)).permute(0,2,1)[...,:3]
    return wc

def make_scatter_img(imgs,ground_pos,res):
    import torch_scatter
    pc_idx = ( (ground_pos*.5+.5).clip(0,1)*(res-1)).long()
    lin_pc_idx=(pc_idx[...,1]*res+pc_idx[...,0])[...,None].long().squeeze(-1).squeeze(0)
    scatter_img = torch_scatter.scatter_mean(imgs[0],lin_pc_idx,0,dim_size=res**2)
    return scatter_img


def softmax_nd_blend( colors, areas,fragments, blend_params, z_background: float, znear: float
        = 1.0, zfar: float = 100,):

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32,
        device=device)

    eps = 1e-10
    # Custom - adding segmentation mask to valid face mask
    mask = (fragments.pix_to_face >= 0).float() #* (colors[...,0]).float()

    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma)*mask
    alpha = torch.prod((1.0 - prob_map), dim=-1)
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None]

    weights_num = prob_map * torch.exp((z_inv - z_inv_max) /
            blend_params.gamma)
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    denom = weights_num.sum(dim=-1)[..., None] + delta

    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    pixel_colors = weighted_colors / denom
    weighted_areas = (weights_num[..., None] * areas).sum(dim=-2)
    pixel_areas = weighted_areas / denom

    return torch.cat((pixel_colors,fragments.zbuf[...,:1]),-1)

class NDShader(torch.nn.Module):
    def __init__(
            self, device="cpu", cameras=None, blend_params=None,
            z_background=100.
            ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.z_background = z_background

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                    or in the forward pass"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params",
                self.blend_params)

        faces = meshes.faces_packed()
        verts = meshes.verts_packed()
        areas, _ = mesh_face_areas_normals(verts, faces)
        areas_pix=interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords,
                areas[:,None,None].expand(-1,3,3)
                )
        images = softmax_nd_blend(texels,areas_pix, fragments, blend_params, 0)
        return images

# Creates pairs from i->i+1 for all i on the second dim
def pairs(src):
    return torch.stack([src [:,[i,i+1]] for i in range(src.size(1)-1)],1)

def cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,) :
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return R_pytorch3d, T_pytorch3d, focal_pytorch3d, p0_pytorch3d, image_size,


class AlphaCompositor(torch.nn.Module):
    """
    Accumulate points using alpha compositing.
    """

    def __init__(
        self, background_color= None
    ) -> None:
        super().__init__()
        self.background_color = background_color

    def forward(self, fragments, alphas, ptclds, **kwargs) -> torch.Tensor:
        background_color = kwargs.get("background_color", self.background_color)
        from pytorch3d.renderer.compositing import alpha_composite
        images = alpha_composite(fragments, alphas, ptclds)

        # images are of shape (N, C, H, W)
        # check for background color & feature size C (C=4 indicates rgba)
        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images
class PointsRenderer(torch.nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        depths=fragments.zbuf[...,0]
        nopix=depths==-1
        depths=torch.where(nopix,-torch.ones_like(depths)*1e3,depths)

        return torch.cat((images,~nopix[...,None],depths[...,None]),-1)
        #return torch.cat((images,~nopix[...,None],depths[...,None]),-1)

def pc_render(wc,feats,cameras,imsize=256,mod=False,save=False,
                #radius=1.5*1e-2):
                radius=3*1e-2):
    if mod:
        wc=torch.cat((-wc[...,[0]],wc[...,[1]],wc[...,[2]]),-1)
    point_cloud = Pointclouds(points=wc[...,:3],features=feats)
    if save:
        print("saving pc")
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
        fig=plot_scene({ "Pointcloud": { "person": point_cloud } })
        torch.save(fig,"/home/camsmith/img/pc.pt")
    raster_settings = PointsRasterizationSettings( image_size=imsize, radius = radius, points_per_pixel = 10, bin_size = None,)
    rasterizer = PointsRasterizer(raster_settings=raster_settings)
    pc_renderer = PointsRenderer( rasterizer=rasterizer, compositor=AlphaCompositor(),)
    if isinstance( cameras, list ):
        return [pc_renderer(point_cloud,cameras=cam) for cam in cameras]
    return pc_renderer(point_cloud,cameras=cameras)
# Crds is bbox as xmin,xmax,ymin,ymax
def get_affine_transf(crd,crop_sl):
    for i,(xmin,xmax,ymin,ymax) in enumerate(crd): # catch case where bbox is single row/col
        if xmin==xmax or ymin==ymax:
            crd[i]=torch.tensor([0,127,0,127])
    points_src = torch.stack([torch.stack([crd[:,0],crd[:,2]]),torch.stack([crd[:,1],crd[:,2]]),
                              torch.stack([crd[:,1],crd[:,3]]),torch.stack([crd[:,0],crd[:,3]])]).permute(2,0,1)
    dst_h=dst_w = crop_sl
    points_dst = torch.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]], device="cuda", dtype=torch.float32
    ).expand(points_src.shape[0], -1, -1)
    return kornia.get_perspective_transform(points_src.cuda().float(), points_dst.cuda().float())

def normalize_images(images):
    images = images[:, :, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
    return (images/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

# Converts pose to lietorch pose
def pose_to_lietorch(poses):
    quat = pytorch3d.transforms.matrix_to_quaternion(poses[...,:3, :3])
    quat = torch.cat((quat[...,1:], quat[...,:1]), -1)
    vec=torch.cat((poses[...,:3,3],quat),-1)
    return SE3.InitFromVec(vec)

# Maps seg mask to bbox
def get_bbox(seg):
    if len(seg.shape)>2:
        return torch.stack([get_bbox(x) for x in seg])
    valid = seg.nonzero()
    if valid.size(0)==0:
        valid=torch.zeros(3,2).cuda()
    return torch.tensor([valid[:,1].min(),valid[:,1].max(),valid[:,0].min(),valid[:,0].max()])

# Connected component extraction, returns n largest cc
# Returns seg masks, bbox crds
def conn_comp(src,n=0):

    segs,bboxs,ccs=[],[],[]
        
    for src_ in src.to("cuda", torch.uint8):
        cc = connected_components_labeling(src_)
        unique = cc.flatten()[src_.flatten().bool()].unique(return_counts=True)

        idx_counts = sorted(list(zip(list(unique[1]),list(unique[0]))),reverse=True)

        if n: idx_counts=idx_counts[:n]

        idxs = [x[1] for x in idx_counts]
        
        #seg = cc == unique[0][unique[1].max(0)[1]]
        ccs.append(cc)
        tmp=[],[]
        if len(idxs)==0: # if cc is all empty, just return whole image as seg
            cc = torch.ones_like(cc)
            idxs=[1]
        for idx in idxs:
            tmp[0].append(cc==idx)
            tmp[1].append(get_bbox(tmp[0][-1]))
        segs.append (torch.stack(tmp[0]))
        bboxs.append(torch.stack(tmp[1]))

    return torch.stack(ccs),segs,bboxs

def get_fg_masks(gt,sl=None):
    b,ctx=gt["context"]["rgb"].shape[:2]

    projs = geometry.obj_projections(gt,sl=sl)

    #bgflowdiff = (projs["flow2d"]-projs["bgflow2d"]).norm(dim=-2).unflatten(-1,(sl,sl)).flatten(0,1)
    #return (bgflowdiff>.2)
    bgflowdiff = (projs["flow3d"]-projs["bgflow3d"]).norm(dim=-2).unflatten(-1,(sl,sl)).flatten(0,1)
    return (bgflowdiff>.01)

# Connected components of camera-independent moving pixels
def fg_conn_comp(gt,sl=None):

    return conn_comp(get_fg_masks(gt,sl))

# Duplicate detection from connected component splits
# Loopy version - can be vectorized later
def detected_duplicates(gt,ccs,segs,sl=None):

    b,ctx=gt["context"]["rgb"].shape[:2]
    context_sl=int(gt["context"]["rgb"].size(-2)**.5)

    ccs=ccs.unflatten(0,(b,ctx-1))

    seg_dups=[None]*len(segs)
    for i in range(ctx-1):
        flows=geometry.obj_projections(gt,sl,ii=torch.ones(ctx-1).long()*i,jj=torch.arange(ctx-1))["2dproj"]
        for j in range(b):

            proj=F.grid_sample(ccs[j][:,None].float(),2*flows[j].permute(0,2,1)[:,None]/context_sl-1,mode="nearest")
            seg=segs[j*(ctx-1)+i]
            dups=[]
            for seg_ in seg:
                dup=False
                for proj_ in proj.squeeze():
                    unique,counts=(seg_.flatten()*proj_).unique(return_counts=True)
                    if len(unique[counts>10])>2:
                        dup=True
                dups.append(dup)
            seg_dups[j*(ctx-1)+i]=dups
    return seg_dups

# Calculates difference in attention map through projection
# Loopy version - can be vectorized later
def temporal_attn_diff_(gt,attn,segs,seg_dups,sl=None):

    b,ctx=gt["context"]["rgb"].shape[:2]
    attn = attn.unflatten(0,(b,ctx-1))
    sl=attn.size(-1)
    context_sl=int(gt["context"]["rgb"].size(-2)**.5)
    rgbs = gt["context"]["rgb"][:,:-1]

    fg_masks = get_fg_masks(gt,sl)

    dup_ims=[]
    for seg,dup in zip(segs,seg_dups):
        all_im = torch.zeros_like(segs[0][0])
        
        badseg = seg[dup]
        for seg_ in badseg:
            all_im = all_im+seg_
        dup_ims.append(all_im)
    dup_ims = torch.stack(dup_ims).unflatten(0,(b,ctx-1))

    dup_ims=torch.stack([sum([x_*y_ for x_,y_ in zip(x,y)]) for x,y in zip(segs,seg_dups)]).unflatten(0,(b,ctx-1))

    consist_masks,err_ims,errs=[],[],0
    for i in range(ctx-1):
        proj_pairs=geometry.obj_projections(gt,sl,ii=torch.ones(ctx-1).long()*i,jj=torch.arange(ctx-1))
        for j in range(b):

            consist_mask = ((proj_pairs["proj3d"][j]-proj_pairs["camcrd"][j][:-1]).norm(dim=1).unflatten(-1,(sl,sl))[:,None]<.8).float()
            fg_mask = fg_masks.unflatten(0,(b,ctx-1))[j][i][None,None].float()

            src_rgb = rgbs[j][i]

            proj=F.grid_sample(attn[j],2*proj_pairs["2dproj"][j].permute(0,2,1).unflatten(1,(sl,sl))/(context_sl-1)-1,mode="nearest")
            dup_proj=F.grid_sample((~dup_ims[j].bool()).float()[:,None],
                                2*proj_pairs["2dproj"][j].permute(0,2,1).unflatten(1,(sl,sl))/(context_sl-1)-1,mode="nearest")
            err = (proj-attn[j][i][None]).abs().sum(1,keepdim=True)*consist_mask*fg_mask*dup_proj
            imsl=int(gt["context"]["rgb"].size(2)**(1/2))
            errs += err.mean()
            err_ims.append(err*F.interpolate(src_rgb.T[None].unflatten(-1,(imsl,imsl)),(sl,sl)))
            consist_masks.append(consist_mask)

    return torch.stack(consist_masks),torch.stack(err_ims),errs


def temporal_attn_diff_(gt,attn,segs,seg_dups,sl=None):

    imsl=256
    b,ctx=gt["context"]["rgb"].shape[:2]
    #attn = attn.unflatten(0,(b,ctx))
    attn = attn.unflatten(-1,(imsl,imsl))
    sl=attn.size(-1)
    context_sl=int(gt["context"]["rgb"].size(-2)**.5)
    rgbs = gt["context"]["rgb"]

    #fg_masks = get_fg_masks(gt,sl)

    #dup_ims=[]
    #for seg,dup in zip(segs,seg_dups):
    #    all_im = torch.zeros_like(segs[0][0])
    #    
    #    badseg = seg[dup]
    #    for seg_ in badseg:
    #        all_im = all_im+seg_
    #    dup_ims.append(all_im)
    #dup_ims = torch.stack(dup_ims).unflatten(0,(b,ctx-1))

    #dup_ims=torch.stack([sum([x_*y_ for x_,y_ in zip(x,y)]) for x,y in zip(segs,seg_dups)]).unflatten(0,(b,ctx-1))

    consist_masks,err_ims,errs=[],[],0
    for i in range(ctx):
        proj_pairs=geometry.obj_projections(gt,sl,ii=torch.ones(ctx).long()*i,jj=torch.arange(ctx))
        for j in range(b):

            #consist_mask = ((proj_pairs["proj3d"][j]-proj_pairs["camcrd"][j][:]).norm(dim=1).unflatten(-1,(sl,sl))[:,None]<.8).float()

            src_rgb = rgbs[j][i]

            proj=F.grid_sample(attn[j],2*proj_pairs["2dproj"][j].permute(0,2,1).unflatten(1,(sl,sl))/(context_sl-1)-1,mode="nearest")
            err = (proj-attn[j][i][None]).abs().sum(1,keepdim=True)#*consist_mask
            imsl=int(gt["context"]["rgb"].size(2)**(1/2))
            errs += err.mean()
            err_ims.append(err*F.interpolate(src_rgb.T[None].unflatten(-1,(imsl,imsl)),(sl,sl)))
            #consist_masks.append(consist_mask)

    #return torch.stack(consist_masks),torch.stack(err_ims),errs
    return errs

def temporal_attn_diff(gt,attn,segs,seg_dups,sl=None):

    imsl=256
    b,ctx=gt["context"]["rgb"].shape[:2]
    #attn = attn.unflatten(0,(b,ctx))
    attn = attn.unflatten(-1,(imsl,imsl))
    sl=attn.size(-1)
    context_sl=int(gt["context"]["rgb"].size(-2)**.5)
    rgbs = gt["context"]["rgb"]

    projs = geometry.obj_projections(gt,highsl)
    b,ctx=gt["context"]["rgb"].shape[:2]

    gtflow=projs["flow2d"].unflatten(-1,(highsl,highsl))
    validmask=projs["validmask"].unflatten(-1,(highsl,highsl))
    gtflow=torch.where(validmask.expand(-1,-1,2,-1,-1),gtflow,torch.zeros_like(gtflow))




    consist_masks,err_ims,errs=[],[],0
    for i in range(ctx):
        proj_pairs=geometry.obj_projections(gt,sl,ii=torch.ones(ctx).long()*i,jj=torch.arange(ctx))
        for j in range(b):

            #consist_mask = ((proj_pairs["proj3d"][j]-proj_pairs["camcrd"][j][:]).norm(dim=1).unflatten(-1,(sl,sl))[:,None]<.8).float()

            src_rgb = rgbs[j][i]

            proj=F.grid_sample(attn[j],2*proj_pairs["2dproj"][j].permute(0,2,1).unflatten(1,(sl,sl))/(context_sl-1)-1,mode="nearest")
            err = (proj-attn[j][i][None]).abs().sum(1,keepdim=True)#*consist_mask
            imsl=int(gt["context"]["rgb"].size(2)**(1/2))
            errs += err.mean()
            err_ims.append(err*F.interpolate(src_rgb.T[None].unflatten(-1,(imsl,imsl)),(sl,sl)))
            #consist_masks.append(consist_mask)

    #return torch.stack(consist_masks),torch.stack(err_ims),errs
    return errs

def analytical_depth(phis,coords,world2model,cam2world):
    with torch.enable_grad():
        depth_infos = [light_field_depth_map(coord,cam2world,phi)
                for phi,coord in zip(phis,coords)]

    mod_xyz    = torch.stack([di["points"] for di in depth_infos])
    mod_xyzh   = torch.cat((mod_xyz,torch.ones_like(mod_xyz[...,:1])),-1)
    world_xyz  = (world2model.inverse()@mod_xyzh.permute(0,1,3,2)
                                                      ).permute(0,1,3,2)[...,:3]
    valid_mask = torch.stack([di["mask"].view(di["depth"].shape) for di in depth_infos])
    est_depth = (1e-4+(geometry.get_ray_origin(cam2world)[None,:,None]-world_xyz
                    ).square().sum(-1,True)).sqrt()*valid_mask
    return est_depth,valid_mask

def get_context_cam(input):
    query_dict = input['context']
    pose = flatten_first_two(query_dict["cam2world"])
    intrinsics = flatten_first_two(query_dict["intrinsics"])
    uv = flatten_first_two(query_dict["uv"].float())
    return pose, intrinsics, uv

def get_query_cam(input):
    query_dict = input['query']
    pose = flatten_first_two(query_dict["cam2world"])
    intrinsics = flatten_first_two(query_dict["intrinsics"])
    uv = flatten_first_two(query_dict["uv"].float())
    return pose, intrinsics, uv

def get_latest_file(root_dir):
    """Returns path to latest file in a directory."""
    list_of_files = glob.glob(os.path.join(root_dir, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def parse_comma_separated_integers(string):
    return list(map(int, string.split(',')))


def scale_img(img, type):
    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img


def convert_image(img, type):
    '''Expects single batch dimesion'''
    img = img.squeeze(0)

    if not 'normal' in type:
        img = detach_all(lin2img(img, mode='np'))

    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img


def write_img(img, path):
    print(img.shape)
    img = lin2img(img)[0]

    img += 1
    img /= 2.
    img = img.detach().cpu().numpy()
    img = np.clip(img, 0., 1.)
    img *= 255

    cv2.imwrite(path, img.astype(np.uint8))


def in_out_to_param_count(in_out_tuples):
    return np.sum([np.prod(in_out) + in_out[-1] for in_out in in_out_tuples])


def flatten_first_two(tensor):
    b, s, *rest = tensor.shape
    return tensor.view(b*s, *rest)


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        line1 = list(map(float, file.readline().split()))
        if line1[-1]==0:
            f, cx, cy, _ = line1
            fy=f
        else:
            f, fy, cx, cy, = line1
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f
        fy= trgt_sidelength / width * fy

    fx = f
    if invert_y:
        fy = -fy

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses

def num_divisible_by_2(number):
    i = 0
    while not number%2:
        number = number // 2
        i += 1

    return i

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pose(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d"%params)

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def encoder_load(model, path):
    if os.path.isdir(path):
        checkpoint_path = sorted(glob(os.path.join(path, "*.pth")))[-1]
    else:
        checkpoint_path = path

    whole_dict = torch.load(checkpoint_path)

    state = model.state_dict()
    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in whole_dict["model"].items() if "encoder" in k}
    # 2. overwrite entries in the existing state dict
    state.update(filtered_dict)
    # 3. load the new state dict
    model.load_state_dict(state)

def custom_load(model, path, discriminator=None, 
                            gen_optimizer=None,disc_optimizer=None):

    whole_dict = torch.load(path)
    model.load_state_dict(whole_dict,strict=False)


def custom_save(model, path, discriminator=None, gen_optimizer=None,disc_optimizer=None):
    torch.save(model.state_dict(), path)

def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(dict_to_gpu(k) for k in ob)
    elif isinstance(ob, list):
        return [dict_to_gpu(k) for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob


def add_batch_dim_to_dict(ob):
    if isinstance(ob, collections.Mapping):
        return {k: add_batch_dim_to_dict(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(add_batch_dim_to_dict(k) for k in ob)
    elif isinstance(ob, list):
        return [add_batch_dim_to_dict(k) for k in ob]
    else:
        try:
            return ob[None, ...]
        except:
            return ob


def detach_all(tensor):
    return tensor.detach().cpu().numpy()


def lin2img(tensor, image_resolution=None, mode='torch'):
    if len(tensor.shape) == 3:
        batch_size, num_samples, channels = tensor.shape
    elif len(tensor.shape) == 2:
        num_samples, channels = tensor.shape

    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    if len(tensor.shape)==3:
        if mode == 'torch':
            tensor = tensor.permute(0, 2, 1).view(batch_size, channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(batch_size, height, width, channels)
    elif len(tensor.shape) == 2:
        if mode == 'torch':
            tensor = tensor.permute(1, 0).view(channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(height, width, channels)

    return tensor


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def parse_intrinsics_hdf5(raw_data, trgt_sidelength=None, invert_y=False):
    s = raw_data[...].tostring()
    s = s.decode('utf-8')

    lines = s.split('\n')

    f, cx, cy, _ = map(float, lines[0].split())
    grid_barycenter = torch.Tensor(list(map(float, lines[1].split())))
    height, width = map(float, lines[3].split())

    try:
        world2cam_poses = int(lines[4])
    except ValueError:
        world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, world2cam_poses


def get_mgrid(sidelen, dim=2, flatten=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.from_numpy(pixel_coords)

    if flatten:
        pixel_coords = pixel_coords.view(-1, dim)
    return pixel_coords


def Nv(st, x, x_prim, d):
    a = x + st[..., :1] * d
    b = x_prim + st[..., 1:] * d
    v_dir = b - a
    v_mom = torch.cross(a, b, dim=-1)
    return torch.cat((v_dir, v_mom), dim=-1) / (v_dir.norm(dim=-1, keepdim=True) + 1e-9)


def horizontal_plucker_slices_thirds(cam2world, light_field_fn, sl=256):
    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]

    slices = []
    sts = []
    s = torch.linspace(-0.5, 0.5, 128)
    t = torch.linspace(-0.5, 0.5, 1024)
    st = torch.stack(torch.meshgrid(s, t), dim=-1).cuda().requires_grad_(True)
    for j, third in enumerate([0.0]):
    # for j, third in enumerate([-0.2, 0.05, 0.2]):
        at = cam2world[:1, ..., :3, 2] + cam2world[:1, ..., :3, 1] * third

        x_prim = x + at
        with torch.enable_grad():
            # st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True) * 0.5
            v_norm = Nv(st, x, x_prim, right)
            reg_model_out = light_field_fn(v_norm)
            slices.append(reg_model_out)
            sts.append(st)

    return {'slices':slices, 'st':sts, 'coords':v_norm}


def lumigraph_slice(cam2world, intrinsics, uv, light_field_fn, sl, row, parallax=0.2):
    uv_img = lin2img(uv[:1], mode='np')
    uv_slice = uv_img[:, row]
    # unproject points
    lift = geometry.lift(uv_slice[..., 0], uv_slice[..., 1], torch.ones_like(uv_slice[..., 0]), intrinsics)

    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]
    at = torch.einsum('...ij,...j', cam2world[0, :, :3, :3], lift[:, lift.shape[1]//2])
    at = F.normalize(at, dim=-1)

    s = torch.linspace(0, parallax, sl).cuda()
    t = torch.nn.Upsample(size=sl, mode='linear', align_corners=True)(lift.permute(0, 2, 1)[:, :1])[0,0]

    x_prim = x + at
    with torch.enable_grad():
        st = torch.stack(torch.meshgrid(s, t), dim=-1).cuda()
        st[..., 1] += torch.linspace(0., parallax, sl)[:, None].cuda()
        st = st.requires_grad_(True)
        v_norm = Nv(st, x, x_prim, right)
        reg_model_out = light_field_fn(v_norm)

    return {'slice':reg_model_out, 'st':st}


def vertical_plucker_slices_thirds(cam2world, light_field_fn, sl=256):
    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]
    down = cam2world[:1, ..., :3, 1]

    slices = []
    s = torch.linspace(-0.5, 0.5, 128)
    t = torch.linspace(-0.5, 0.5, 1024)
    st = torch.stack(torch.meshgrid(s, t), dim=-1).cuda().requires_grad_(True)
    for j, third in enumerate([0.]):
    # for j, third in enumerate([-0.15, 0., 0.15]):
        at = cam2world[:1, ..., :3, 2] + right * third

        x_prim = x + at
        with torch.enable_grad():
            # st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True) * 0.5
            v_norm = Nv(st, x, x_prim, down)
            reg_model_out = light_field_fn(v_norm)
            slices.append(reg_model_out)

    return {'slices':slices, 'st':st, 'coords':v_norm}


def get_view_grid(cam2world, grid_sl, offset=1):
    right = cam2world[:1, ..., :3, 0]
    down = cam2world[:1, ..., :3, 1]

    view_grid = []
    for row in np.linspace(1, -1, grid_sl):
        row_list = []
        for col in np.linspace(1, -1, grid_sl):
            new_cam2world = cam2world.clone()
            new_cam2world[..., :3, 3] += row * offset * down + col * offset * right
            row_list.append(new_cam2world)
        view_grid.append(row_list)
    return view_grid


def canonical_plucker_slice(cam2world, light_field_fn, sl=256):
    x = geometry.get_ray_origin(cam2world)[:1]
    right = cam2world[:1, ..., :3, 0]
    at = cam2world[:1, ..., :3, 2]

    x_prim = x + at
    with torch.enable_grad():
        st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True) * 0.5
        v_norm = Nv(st, x, x_prim, right)
        reg_model_out = light_field_fn(v_norm)

    return {'slice':reg_model_out, 'st':st, 'coords':v_norm}


def plucker_slice(ray_origin, right, at, light_field_fn, sl=256):
    plucker = geometry.plucker_embedding(cam2world, uv, intrinsics)
    right = cam2world[:1, ..., :3, 0]
    at = cam2world[:1, ..., :3, 2]

    x = geometry.get_ray_origin(cam2world)[:1]

    intersections = geometry.lift(uv[...,0], uv[...,1], torch.ones_like(uv[...,0]), intrinsics=intrinsics)
    s = intersections[0, ..., 0]
    t = torch.linspace(-1, 1, s.shape[0]).cuda()

    x_prim = x + at
    with torch.enable_grad():
        st = torch.stack(torch.meshgrid(s, t), dim=-1).requires_grad_(True).cuda()

        a = x + plucker[..., :3] + st[..., :1] * right
        b = x_prim + st[..., 1:] * right
        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / (v_dir.norm(dim=-1, keepdim=True) + 1e-9)
        reg_model_out = light_field_fn(v_norm)

    return {'slice':reg_model_out, 'st':st, 'coords':v_norm}


def get_random_slices(light_field_fn, k=10, sl=128):
    x = torch.zeros((k, 1, 3)).cuda()
    x_prim = torch.randn_like(x).cuda()
    x_prim = F.normalize(x_prim, dim=-1)

    d = torch.normal(torch.zeros_like(x), torch.ones_like(x)).cuda()
    d = F.normalize(d, dim=-1)

    with torch.enable_grad():
        st = get_mgrid(sl).cuda().flatten(1, 2).requires_grad_(True)
        coords = Nv(st, x, x_prim, d)
        c = light_field_fn(coords)

    return {'slice':c, 'st':st, 'coords':coords}


def light_field_point_cloud(light_field_fn, num_samples=64**2, outlier_rejection=True):
    dirs = torch.normal(torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)).cuda()
    dirs = F.normalize(dirs, dim=-1)

    x = (torch.rand_like(dirs) - 0.5) * 2

    D = 1
    x_prim = x + D * dirs

    st = torch.zeros(1, num_samples, 2).requires_grad_(True).cuda()
    max_norm_dcdst = torch.ones_like(st) * 0
    dcdsts = []
    for i in range(5):
        d_prim = torch.normal(torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)).cuda()
        # d_prim = F.normalize(torch.cross(d_prim, dirs, dim=-1))
        # d_prim += torch.normal(torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)).cuda() * 1e-3
        d_prim = F.normalize(d_prim, dim=-1)

        a = x + st[..., :1] * d_prim
        b = x_prim + st[..., 1:] * d_prim
        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / v_dir.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            c = light_field_fn(v_norm)
            dcdst = diff_operators.gradient(c, st)
            dcdsts.append(dcdst)
            criterion = max_norm_dcdst.norm(dim=-1, keepdim=True)<dcdst.norm(dim=-1, keepdim=True)
            # dir_dot = torch.abs(torch.einsum('...j,...j', d_prim, dirs))[..., None]
            # criterion = torch.logical_and(criterion, dir_dot<0.1)
            max_norm_dcdst = torch.where(criterion, dcdst, max_norm_dcdst)

    dcdsts = torch.stack(dcdsts, dim=0)
    dcdt = dcdsts[..., 1:]
    dcds = dcdsts[..., :1]

    d = D * dcdt / (dcds + dcdt)
    mask = d.std(dim=0) > 1e-2
    d = d.mean(0)
    d[mask] = 0.
    d[max_norm_dcdst.norm(dim=-1)<1] = 0.

    # if outlier_rejection:

    return {'depth':d, 'points':x + d * dirs, 'colors':c}


def get_pencil_dirs(plucker_coords, cam2world, light_field_fn):
    x = geometry.get_ray_origin(cam2world)
    at = cam2world[..., :3, 2]
    right = cam2world[..., :3, 0]
    x_prim = x + at

    st = torch.zeros_like(plucker_coords[..., :2]).requires_grad_(True).to(plucker_coords.device)
    # d_prim = torch.normal(torch.zeros_like(plucker_coords[..., :3]), torch.ones_like(plucker_coords[..., :3])).to(plucker_coords.device)
    # d_prim = F.normalize(d_prim, dim=-1)
    # d_prim = torch.normal(torch.zeros(1, 1, 3), torch.ones(1, 1, 3)).to(plucker_coords.device)
    # d_prim = F.normalize(d_prim, dim=-1)
    d_prim = right

    with torch.enable_grad():
        c = light_field_fn(Nv(st, x, x_prim, d_prim))
        dcdst = diff_operators.gradient(c, st)

    confidence = dcdst.norm(dim=-1, keepdim=True)

    dcdst = F.normalize(dcdst, dim=-1)
    J = torch.Tensor([[0, -1], [1, 0.]]).cuda()
    rot_grad = torch.einsum('ij,bcj->bci', J, dcdst)

    dcdt = dcdst[..., 1:]
    dcds = dcdst[..., :1]

    def pencil(a):
        return light_field_fn(Nv(st+a*rot_grad, x, x_prim, d_prim))

    return {'confidence':confidence, 'pencil_dir':rot_grad, 'pencil_fn':pencil}


def get_canonical_pencil_dirs(plucker_coords, light_field_fn):
    x = geometry.get_ray_origin(cam2world)
    right = cam2world[..., :3, 0]
    at = cam2world[..., :3, 2]

    x_prim = x + at
    st = torch.zeros_like(plucker_coords[..., :2]).requires_grad_(True).to(plucker_coords.device)
    # d_prim = torch.normal(torch.zeros_like(plucker_coords[..., :3]), torch.ones_like(plucker_coords[..., :3])).to(plucker_coords.device)
    # d_prim = F.normalize(d_prim, dim=-1)

    with torch.enable_grad():
        c = light_field_fn(Nv(st, x, x_prim, right))
        dcdst = diff_operators.gradient(c, st)

    J = torch.Tensor([[0, -1], [1, 0.]]).cuda()
    rot_grad = torch.einsum('ij,bcj->bci', J, dcdst)

    dcdt = dcdst[..., 1:]
    dcds = dcdst[..., :1]

    return {'confidence':torch.abs(dcds + dcdt), 'pencil_dir':rot_grad}


def depth_map(query):
    light_field_fn = model.get_light_field_function(query['z'])

    plucker_coords = geometry.plucker_embedding(cam2world, uv, intrinsics)
    return light_field_depth_map(plucker_coords, cam2world, light_field_fn)

def light_field_depth_map(plucker_coords, cam2world, light_field_fn,niter=4):
    x = geometry.get_ray_origin(cam2world)[:,None]
    D = 1
    x_prim = x + D * plucker_coords[..., :3]

    d_prim = torch.normal(torch.zeros_like(plucker_coords[..., :3]),
          torch.ones_like(plucker_coords[..., :3])).to( plucker_coords.device)
    d_prim = F.normalize(d_prim, dim=-1)

    dcdsts = []
    for i in range(niter):
        st = ((torch.rand_like(plucker_coords[..., :2]) - 0.5) * 1e-2).requires_grad_(True).to(plucker_coords.device)
        a = x + st[..., :1] * d_prim
        b = x_prim + st[..., 1:] * d_prim

        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / v_dir.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            c = light_field_fn(v_norm)
            dcdst = diff_operators.gradient(c, st, create_graph=False)
            dcdsts.append(dcdst)
            del dcdst
            del c

    dcdsts = torch.stack(dcdsts, dim=0)

    dcdt = dcdsts[0, ..., 1:]
    dcds = dcdsts[0, ..., :1]

    all_depth_estimates = D * dcdsts[..., 1:] / (dcdsts.sum(dim=-1, keepdim=True))
    all_depth_estimates[torch.abs(dcdsts.sum(dim=-1)) < 5] = 0
    all_depth_estimates[all_depth_estimates<0] = 0.

    depth_var = torch.std(all_depth_estimates, dim=0, keepdim=True)

    d = D * dcdt / (dcds + dcdt)
    invalid = (
               (torch.abs(dcds + dcdt) < 5).flatten()|
               (d<0).flatten()|
               (depth_var[0, ..., 0] > 0.01).flatten()
               )
    d[invalid.view(d.shape)] = 0.
    return {'depth':d, 'points':x + d * plucker_coords[..., :3],"mask":~invalid}

def assemble_model_input(context, query, gpu=True):
    context['mask'] = torch.Tensor([1.])
    query['mask'] = torch.Tensor([1.])

    context = add_batch_dim_to_dict(context)
    context = add_batch_dim_to_dict(context)

    query = add_batch_dim_to_dict(query)
    query = add_batch_dim_to_dict(query)

    model_input = {'context': context, 'query': query, 'post_input': query}

    if gpu:
        model_input = dict_to_gpu(model_input)
    return model_input


def grads2img(mG):
    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return mGrgb


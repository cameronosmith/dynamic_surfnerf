import cv2
import os
import torch.nn.functional as F
import torch
import random
import imageio
import numpy as np
from glob import glob
from collections import defaultdict
import skimage.filters
from pdb import set_trace as pdb
from itertools import combinations
from random import choice
import matplotlib.pyplot as plt

from torchvision import transforms

# Geometry functions below used for calculating depth, ignore
def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs
def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]

def parse_intrinsics(intrinsics):
    fx = intrinsics[..., 0, :1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    return fx, fy, cx, cy

hom = lambda x,i=-1: torch.cat((x,torch.ones_like(x.unbind(i)[0].unsqueeze(i))),i)
def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x

def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)

def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)
    world_coords = torch.einsum('b...ij,b...kj->b...ki', cam2world, pixel_points_cam)[..., :3]

    return world_coords


def get_ray_directions(xy, cam2world, intrinsics,normalize=True):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    if normalize:ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs


class SceneInstanceDataset(torch.utils.data.Dataset):
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,
                 input_img_sidelength=None,
                 img_sidelength=None,
                 num_images=None,
                 use_seg=False,
                 num_seg=-1,
                 use_depth=False,
                 pose2=False,
                 cache=None,
                 raft=None,
                    ):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.input_img_sidelength = input_img_sidelength
        self.instance_dir = instance_dir
        self.cache = {}

        self.interp_mode = cv2.INTER_LINEAR

        pose_dir  = os.path.join(instance_dir, "pose")
        color_dir = os.path.join(instance_dir, "image")

        import pykitti
        drive=self.instance_dir.strip("/").split("/")[-1].split("_")[-2]
        date=self.instance_dir.strip("/").split("/")[-2]
        self.kitti_raw=pykitti.raw( "/".join( self.instance_dir.rstrip("/").split("/")[:-2] ), date,drive)
        self.num_img = len(os.listdir(os.path.join(self.instance_dir,self.instance_dir,"image_02/data")))

        self.disps = sorted(glob(os.path.join(self.instance_dir,"disps","*.npy")))

        self.color_paths = sorted(glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))
        self.instance_name = os.path.basename(os.path.dirname(self.instance_dir))

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths  = pick(self.pose_paths, specific_observation_idcs)
        elif num_images is not None:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths  = pick(self.pose_paths, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx, context=False,input_context=True):

        rgb = transforms.ToTensor()(self.kitti_raw.get_cam2(idx))*2-1
            
        disp = torch.from_numpy(np.load(os.path.join(self.instance_dir,"disps",self.disps[idx])))
        zval = self.kitti_raw.calib.K_cam2[0,0]*self.kitti_raw.calib.b_rgb/disp.abs()
        ydiff=(rgb.size(-2)-zval.size(-2))//2
        xdiff=(rgb.size(-1)-zval.size(-1))//2
        zval=zval[-ydiff:ydiff,-xdiff:xdiff]

        K = torch.from_numpy(self.kitti_raw.calib.K_cam2.copy())
        cam2imu = torch.from_numpy(self.kitti_raw.calib.T_cam2_imu).inverse()
        imu2world = torch.from_numpy(self.kitti_raw.oxts[idx].T_w_imu)
        cam2world = (imu2world@cam2imu).float()

        uv = np.mgrid[0:rgb.size(1), 0:rgb.size(2)].astype(float).transpose(1, 2, 0)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        
        # Downsample 
        h,w=rgb.shape[-2:]
        lowh,loww=128,416
        rgb=F.interpolate(rgb[None],(lowh,loww),mode="bilinear",align_corners=True)[0]
        zval=F.interpolate(zval[None,None],(lowh,loww),mode="bilinear",align_corners=True)[0,0]
        K = torch.stack((K[0]/w,K[1]/h,K[2]))
        uv = np.mgrid[0:lowh, 0:loww].astype(float).transpose(1, 2, 0)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        uv = uv/torch.tensor([loww,lowh]) # uv in [0,1]

        camcrd = lift(*uv.flatten(0,1)[None].unbind(-1), zval.flatten(0,1)[None],K[None],homogeneous=True)
        wc = ( cam2world[None].float() @ camcrd.permute(0,2,1).float() )[:,:3].permute(0,2,1)
        depth = (cam2world[None][:,:3,-1][:,None] - wc).norm(dim=-1,keepdim=True).squeeze().unflatten(0,(lowh,loww))

        sample = {
            "instance_name": self.instance_name,
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze().long(),
            "cam2world": cam2world,
            "img_idx": torch.Tensor([idx]).squeeze().long(),
            "rgb": rgb,
            "depth": depth,
            "intrinsics": K.float(),
            "uv": uv,
        }

        return sample


def get_instance_datasets(root, max_num_instances=None, specific_observation_idcs=None,
                          cache=None, sidelen=None, max_observations_per_instance=None):
    instance_dirs = sorted(glob(os.path.join(root, "*/")))
    assert (len(instance_dirs) != 0), f"No objects in the directory {root}"

    if max_num_instances != None:
        instance_dirs = instance_dirs[:max_num_instances]

    all_instances = [SceneInstanceDataset(instance_idx=idx, instance_dir=dir,
                      specific_observation_idcs=specific_observation_idcs, img_sidelength=sidelen,
                                          cache=cache, num_images=max_observations_per_instance)
                     for idx, dir in enumerate(instance_dirs)]
    return all_instances


class KittiDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 num_context, num_trgt, root_dir,
                 vary_context_number=False,
                 query_sparsity=None,
                 img_sidelength=None,
                 input_img_sidelength=None,
                 max_num_instances=None,
                 max_observations_per_instance=None,
                 specific_observation_idcs=None,
                 test=False,
                 test_context_idcs=None,
                 context_is_last=False,
                 context_is_first=False,
                 cache=None,
                 video=True,
                 use_seg=False,
                 use_depth=False,
                 pose2=False,
                 num_seg=-1,
                 ):

        basedirs = list(filter(lambda x : "20" in x and "zip" not in x , os.listdir(root_dir)))
        drive_paths=[]
        for basedir in basedirs:
            dirs=list(filter(lambda x: "txt" not in x, os.listdir(os.path.join(root_dir,basedir))))
            drive_paths += [os.path.abspath(os.path.join(root_dir,basedir,dir_)) for dir_ in dirs]
        self.instance_dirs=sorted(drive_paths)

        self.num_context = num_context
        self.num_trgt = num_trgt
        self.query_sparsity = query_sparsity
        self.img_sidelength = img_sidelength
        self.vary_context_number = vary_context_number
        self.cache = {}
        self.test = test
        self.test_context_idcs = test_context_idcs
        self.context_is_last = context_is_last
        self.context_is_first = context_is_first

        print(f"Root dir {root_dir}, {len(self.instance_dirs)} instances")

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        self.max_num_instances=max_num_instances
        if max_num_instances==1:
            self.instance_dirs = [x for x in self.instance_dirs if "2011_09_26_drive_0009_sync" in x];print("note testing single dir") #testing dir

        #subdirs = set([x.split("/")[-2].split("_")[0]+"_0" for x in self.instance_dirs])
        #subdirs = self.instance_dirs
        #self.instance_dirs = [os.path.join(root_dir,subdir) for subdir in subdirs]
        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   input_img_sidelength=input_img_sidelength,
                                                   num_images=max_observations_per_instance,
                                                   cache=cache,
                                                   use_seg=use_seg,
                                                   use_depth=use_depth,
                                                   num_seg=num_seg,
                                                   pose2=pose2,
                                                   )
                              for idx, dir in enumerate(self.instance_dirs)
                                    ]
        self.all_instances = [x for x in self.all_instances if len(x)>40]
        if max_num_instances is not None: self.all_instances = self.all_instances[:max_num_instances]

        # Still need to finish running disparities on all of kitti
        #self.all_instances = [x for x in self.all_instances if len(x.disps)==x.num_img]
        
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def sparsify(self, dict, sparsity):
        new_dict = {}
        if sparsity is None:
            return dict
        else:
            # Sample upper_limit pixel idcs at random.
            rand_idcs = np.random.choice(self.img_sidelength**2, size=sparsity, replace=False)
            for key in ['rgb', 'uv']:
                new_dict[key] = dict[key][rand_idcs]

            for key, v in dict.items():
                if key not in ['rgb', 'uv']:
                    new_dict[key] = dict[key]

            return new_dict

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with which images are loaded."""
        self.img_sidelength = new_img_sidelength
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return int(np.sum(self.num_per_instance_observations))

    def get_instance_idx(self, idx):
        if self.test:
            obj_idx = 0
            while idx >= 0:
                idx -= self.num_per_instance_observations[obj_idx]
                obj_idx += 1
            return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])
        else:
            return np.random.randint(self.num_instances), 0

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys:
                result[key].append(entry[key])

        for key in keys:
            try:
                result[key] = torch.stack(result[key], dim=0)
            except:
                continue

        return result

    def __getitem__(self, idx, sceneidx=None):
        context = []
        trgt = []
        post_input = []

        obj_idx, det_idx = self.get_instance_idx(idx)

        if sceneidx is not None:
            obj_idx, det_idx = sceneidx, sceneidx

        nc = self.num_context
        nq = self.num_trgt

        # Pick a video clip of length 1 + n_qry + 1, use last frame if ctx=2
        assert nc <= 2

        num_skip = 3
        #randskip = lambda : np.random.randint( 0, len(self.all_instances[obj_idx]) - (nc+nq) * num_skip - 1)
        randskip = lambda : np.random.randint(1, 4)
        first_i = i = randskip()

        if self.max_num_instances:
            first_i=i=0

        # First context
        sample = self.all_instances[obj_idx].__getitem__(
            i, context=True, input_context=True
        )
        context.append(sample)

        # After context comes query frames
        for _ in range(nq):
            sample = self.all_instances[obj_idx].__getitem__(
                i, context=True, input_context=True
            )

            post_input.append(sample)
            post_input[-1]["mask"] = torch.Tensor([1.0])
            sub_sample = self.sparsify(sample, self.query_sparsity)
            trgt.append(sub_sample)
            i = i + randskip()

        # After query frame comes potential second frame as last qry frame
        if nc == 2:
            i = 2#i - randskip()
            sample = self.all_instances[obj_idx].__getitem__(
                i, context=True, input_context=True
            )
            context.append(sample)

        post_input = self.collate_fn(post_input)
        trgt = self.collate_fn(trgt)

        context = self.collate_fn(context)
        out_dict = {"query": trgt, "post_input": post_input, "context": None}, trgt
        out_dict[0]["context"] = context
        return out_dict

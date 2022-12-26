import comet_ml

import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))

from pdb import set_trace as pdb


import torch

import torch.nn as nn

import models 
import summaries
import loss_functions
import custom_layers

import util

from train import train, rapid_test
testing = len(sys.argv)>1 or False
memtest = "mem" in sys.argv
mode=rapid_test if "a" in sys.argv else train
sato = True
sato_cpu=testing and sato

run_large=True

vid = False

model = models.DynamicSurfNerf(backbone="fpn",num_tok=2).cuda()

model = nn.DataParallel(model)

summary_fns = [ summaries.depth,
                summaries.rgb,
                summaries.surface_token,
               ]
delay=-500
loss_fns    = [
               (loss_functions.rgb, 2e1,None),
               #(loss_functions.multiscale_rgb, 1e1,None),
               #(loss_functions.depth, 2e2,None),
               (loss_functions.depth_grad, 1e-1,None),
               (loss_functions.percept, 2e0,None),
               #(loss_functions.peaky_surf, 1e-1,None),
               ]
debug=1
datasets_path =["/om2/user/camsmith/datasets","/nobackup/users/camsmith/datasets"][sato]
logs_dir      =["/om2/user/camsmith/logs",    "/nobackup/users/camsmith/logs"]    [sato]
mode({
        #"data_dir":"/nobackup/users/camsmith/kitti_download/",
        #"val_dir":"/nobackup/users/camsmith/kitti_val/",
        "data_dir":"/nobackup/users/camsmith/kitti_val/",
        "weights_path":"/nobackup/users/camsmith/logs/justrgb_kit_all/checkpoints/00026368.pth",
        #"weights_path":"/nobackup/users/camsmith/logs/oftest_scratch/checkpoints/00000824.pth",
        "batch_size": torch.cuda.device_count(),
        "context_is_last": False,
        "pin_mem":False,
        "comet":not memtest,
        "model":model,
        "val_iters":15,
        "lr":1e-5,
        "update_iter":103,
        "num_context": 2,
        "num_query":5,
        "pose2":False,
        "log_dir":f"{logs_dir}/justrgb_kit_all_additionallosses",
        "num_img":None,
        "num_inst":None,
        "img_sidelength":128,
        "summary_fns":summary_fns,
        "loss_fns":loss_fns,
        "video":vid,
        })

import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import torch.nn.functional as F
import geometry
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from einops import repeat
import util
import torchvision
from einops import rearrange,repeat
import cv2
from pdb import set_trace as pdb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os

def tmp(model_out, gt, writer, iter_):
    rgb(model_out, gt, writer, iter_)
    depth(model_out, gt, writer, iter_)
    surface_token(model_out, gt, writer, iter_)

def write_img(imgs,writer,iter_,title,nrow=8,write_local=True,normalize=True,img_grid=True,value_range=None):
    if img_grid:
        img_grid = torchvision.utils.make_grid(imgs,range=value_range,
            scale_each=False, normalize=normalize,nrow=nrow,pad_value=1).cpu().detach().numpy()
    else:
        img_grid=imgs
    if writer is None and write_local:
        home = os.path.expanduser('~') 
        plt.imsave(f"/nobackup/users/{home}/img/{title}.png",img_grid.astype(float).transpose(1,2,0))
    elif writer is not None:
        writer.add_image(title, img_grid, iter_)


def rgb(model_out, gt, writer, iter_,VAL=""):
    imsl=128
    write_img(model_out["rgb"].clip(-1,1)*.5+.5, writer, iter_, VAL+f"RGB_PRED", normalize=False)
    write_img(model_out["gtrgb"], writer, iter_, VAL+f"RGB_GT", normalize=True)

def surface_token(model_out, gt, writer, iter_,VAL=""):
    imsly,imslx=gt["context"]["rgb"].shape[2:4]
    write_img(model_out["surface_token_idx"]/4, writer, iter_, VAL+f"Surface Token Index", normalize=False,nrow=gt["query"]["uv"].size(1))
def depth(model_out, gt, writer, iter_,VAL=""):
    imsl=128
    depthest=model_out["depth"].clip(3,100)
    write_img(depthest,writer,iter_,"depthest_norm")
    write_img((1/depthest),writer,iter_,"depthest_inv")
    if "gtdepth" in model_out:
        depthgt=model_out["gtdepth"].clip(3,100)
        write_img(depthgt,writer,iter_,"depthgt_norm")
        write_img((1/depthgt),writer,iter_,"depthgt_inv")

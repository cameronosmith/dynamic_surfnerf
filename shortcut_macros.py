import matplotlib.pyplot as plt
import torch
F=torch.nn.functional
from einops import repeat,rearrange

ch_fst = lambda x:rearrange(x,"... (x y) c -> ... c x y",x=int(x.size(-2)**(.5)))
ch_sec = lambda x:rearrange(x,"... c x y -> ... (x y) c")
interp = lambda x,y,mode="bilinear":F.interpolate(x,y,mode=mode,align_corners=True if mode!="nearest" else None)
grid_samp = lambda x,y,pad="zeros",mode="bilinear",align=True: F.grid_sample(x if x.size(0)==y.size(0) or x.size(0)!=1 else repeat(x,"1 ... -> y ...",y=y.size(0)),
                                      y.type(x.type()),mode=mode,align_corners=align,padding_mode=pad)
zeros  = lambda x: torch.zeros_like(x)
ones   = lambda x: torch.ones_like(x)
eye    = lambda x: torch.eye(4).cuda().expand(x,-1,-1)
hom    = lambda x: torch.cat((x,torch.ones_like(x[...,[0]])),-1)

def imsave(x,y=""):
    if x.size(0)==3:
        x=x.permute(1,2,0)
    if len(x.shape)==3 and x.min()<-.1:
        x = x*.5+.5
    plt.imsave("/nobackup/users/camsmith/img/tmp%s.png"%y,x.cpu().numpy())

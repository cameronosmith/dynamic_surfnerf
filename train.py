"""
This file just defines an abstracted infinite training loop
Ignore the rapid test loop if not developing code
"""
import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))
from importlib import reload
from functools import partial
from copy      import deepcopy
import traceback

import gc

import torch
from torch.utils.data        import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from kitti_dataio import KittiDataset as SceneClassDataset

import loss_functions
import summaries
import util,geometry,custom_layers,conv_modules

import comet_ml

from pdb import set_trace as pdb

user = os.path.expanduser("~").split("/")[-1]
project_name = ["slotlfn", "movingslotlfn"][1]
comet_config = {"api_key":"74OhBFkR7Ifli0sEFnTZXAtSQ",
                "project_name":project_name,
                "workspace":"cameronosmith" if user=="camsmith" else "",
                "disabled":0,
                }

# Sets up logging env and returns a writer
def make_writer(log_dir):

    # Setup directories and create writer
    ckpt_dir   = os.path.join(log_dir, 'checkpoints')
    events_dir = os.path.join(log_dir, 'events')
    util.cond_mkdir(log_dir)
    util.cond_mkdir(ckpt_dir)
    util.cond_mkdir(events_dir)
    return ckpt_dir,events_dir,SummaryWriter(events_dir,comet_config=comet_config)

autocast_enable=False

def_opt = {
        "data_dir":None,
        "val_dir":None,
        "batch_size":2,
        "model":None,
        "lr":5e-4,
        "update_iter":1000,
        "weights_path":None,
        "log_dir":"",
        "num_img":None,
        "val_iters":0,
        "num_inst":None,
        "comet":1,
        "num_context":0,
        "summary_fns":[],
        "loss_fns":[],
        }
# Trains a model until user termination
# exp_opt : a dict overriding the defaults above
def train(exp_opt):

    # Unpack experiment script opts
    opt = deepcopy(def_opt)
    for k,v in exp_opt.items(): opt[k] =v

    comet_config["disabled"]=not exp_opt["comet"]

    ckpt_dir,events_dir,writer = make_writer(opt["log_dir"])

    # Load data
    dataset = SceneClassDataset(opt["num_context"], opt["num_query"],
                                root_dir=opt["data_dir"],
                                max_num_instances=opt["num_inst"],)

    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True,
                            drop_last=True, num_workers=4,
                            pin_memory=True)

    if opt["val_dir"] is not None:
        val_dataset = SceneClassDataset(opt["num_context"], opt["num_query"],
                                    root_dir=opt["val_dir"],
                                    max_num_instances=None,)

    print(f"Using logdir {opt['log_dir']}")

    iter_ = 0
    gen_optimizer = torch.optim.Adam(opt["model"].parameters(),lr=opt["lr"])

    if opt["weights_path"] is not None:
        util.custom_load(model=opt["model"],
                gen_optimizer=None,
                path=opt["weights_path"])

    print("manually setting model gan fix ")

    scaler = torch.cuda.amp.GradScaler()
    # Train until user termination

    while True:
        for model_input, ground_truth in dataloader:

            if iter_ and iter_%opt["update_iter"]==0:
                util.custom_save(
                        opt["model"],
                        ckpt_dir+"/%08d.pth"%iter_,
                        gen_optimizer=None,
                        )
                print("wrote")

            model_input,ground_truth = [util.dict_to_gpu(x) for x in
                                                  (model_input,ground_truth)]

            if iter_%20==0 and 1:
                with torch.cuda.amp.autocast(enabled=autocast_enable):
                    with torch.no_grad():
                        if iter_%40==0:
                            #out = opt["model"](model_input,True)
                            with torch.no_grad(): out = util.render_full_img(opt["model"].module,model_input,None)
                            for summary_fn in opt["summary_fns"]:
                                with torch.no_grad():
                                    try:
                                        summary_fn(out,model_input,writer,iter_)
                                    except:
                                        print("skipping summary fn for err",summary_fn)
                                torch.cuda.empty_cache()
                                gc.collect()
                            del out

            # Train gen or disc

            gen_optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=autocast_enable):
                model_output = opt["model"](model_input)
            gen_loss = 0
            for i,(loss_fn,weight,schedule) in enumerate(opt["loss_fns"]):
                start_iter=schedule if schedule is not None else -1
                loss = weight*loss_fn(model_output,ground_truth,model_input)
                loss_active = int(iter_>start_iter)
                writer.add_scalar(loss_fn.__name__, loss.item()*loss_active, iter_)
                gen_loss += loss_active*loss
                print(loss_fn.__name__,loss_active,loss.item())
            if autocast_enable:
                scaler.scale(gen_loss).backward()
                scaler.step(gen_optimizer)
                scaler.update()
            else:
                gen_loss.backward()
                gen_optimizer.step()

            sys.stdout.write("\rIter %06d" % iter_)
            iter_ +=1

        # Validation testing
        if opt["val_dir"] is not None:
            print("val set test")
            val_iters=0

            val_dataloader = DataLoader(val_dataset, batch_size=opt["batch_size"], shuffle=True,
                                drop_last=True, num_workers=0,
                                pin_memory=False)
            for model_input, ground_truth in val_dataloader:

                if val_iters==opt["val_iters"]:
                    print("val iter ",val_iters)
                    break
                val_iters += 1

                with torch.no_grad():
                    model_input,ground_truth = [util.dict_to_gpu(x) for x in
                                                          (model_input,ground_truth)]
                    model_output = opt["model"](model_input)

                    for i,(loss_fn,weight,schedule) in enumerate(opt["loss_fns"]):
                        loss = weight*loss_fn(model_output,ground_truth,model_input)
                        writer.add_scalar("VAL_"+loss_fn.__name__, loss, iter_+val_iters)
                    model_output = opt["model"](model_input,True)

                    if val_iters%5==0:
                        for summary_fn in opt["summary_fns"]:
                            summary_fn(model_output,model_input,writer,iter_+val_iters,"VAL_")

                    sys.stdout.write("\rVal iter %06d" % val_iters)


# Used to quickly test the input/output of a model in the pdb interface instead 
# of training it. Intended usage for any outsiders reading is to edit model in
# other file while fixing errors to avoid model and dataloader creation waiting.
def rapid_test(exp_opt):

    # Unpack experiment script opts and load into namespace (convenient evil)
    opt = deepcopy(def_opt)
    for k,v in exp_opt.items(): opt[k] =v

    # Load data

    print("doing dataloader")
    dataset = SceneClassDataset(opt["num_context"], opt["num_query"],
                                root_dir=opt["data_dir"],
                                video=opt["video"],
                                max_num_instances=opt["num_inst"],
                                context_is_last=opt["context_is_last"],
                                img_sidelength=opt["img_sidelength"],)
    dataset[0];
    print("doing dataloader")
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=False)

    if opt["val_dir"] is not None:
        val_dataset = SceneClassDataset(opt["num_context"], opt["num_query"],
                                    root_dir=opt["val_dir"],
                                    max_num_instances=opt["num_inst"],)
        val_dataloader = DataLoader(val_dataset, batch_size=opt["batch_size"], shuffle=True,
                                drop_last=True, num_workers=1,
                                pin_memory=False)
        val_dataset[0];

    for model_input, gt in dataloader: break
    model_input,gt = [util.dict_to_gpu(x) for x in
                                          (model_input,gt)]
    writer=None

    model=opt["model"]
    
    if opt["weights_path"] is not None:
        util.custom_load(model=model,path=opt["weights_path"])

    log_dataset=False
    rf=1 # render full img
    rc=0 # render custom cam traj
    while True:
        # Reload model's forward and any other fn's, and rerun model with input
        if "DataParallel" in type(model).__name__:
            model.forward=partial(getattr(reload(__import__("models")),
                                            type(model.module).__name__).forward,model.module)
            model.surfnerf_render=partial(getattr(reload(__import__("models")),
                                            type(model.module).__name__).surfnerf_render,model.module)
        else:
            model.forward=partial(getattr(reload(__import__("models")),
                                            type(model).__name__).forward,model)
            model.surfnerf_render=partial(getattr(reload(__import__("models")),
                                            type(model).__name__).surfnerf_render,model)
        print("doing reload")
        for mod in [loss_functions,summaries,geometry,util,
                    custom_layers,conv_modules]:
            reload(mod)
        try: 

            print("doing run")
            with torch.cuda.amp.autocast(enabled=autocast_enable):
                with torch.no_grad(): out = model(model_input)

            loss=loss_functions.tmp(out,gt,model_input)

            if rf:
                print("doing full img")
                with torch.no_grad(): out = util.render_full_img(model,model_input,None)
                summaries.tmp(out,model_input,None,0)

            if rc:
                print("rendering custom traj")
                with torch.no_grad(): out = util.render_cam_traj(model,model_input,None)
            raise Exception("No error")
        except:
            print(traceback.format_exc())
            pdb()

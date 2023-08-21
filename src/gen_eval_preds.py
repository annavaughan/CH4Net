from matplotlib import pyplot as plt
import torch 
import numpy as np 

from models import *
import os
import argparse
from tqdm import tqdm

from models import *
from trainer import *
from loader import *

from torch.utils.data import DataLoader
from glob import glob
import sys

# python3 gen_eval_preds.py 12 final_12_all/ final_12_preds/ 0
in_dir = sys.argv[2]
out_dir = sys.argv[3]
channels = int(sys.argv[1])
alli_yn = bool(int(sys.argv[4]))

print("Loading losses...")
losses = np.load(in_dir+"losses.npy")
best_epoch = np.argmin(losses)

device = torch.device('cuda')

# Set up model
model = Unet(in_channels=channels,
            out_channels=1,
            div_factor=1, 
            prob_output=False)
model = model.to(device)
model = nn.DataParallel(model)

model.load_state_dict(torch.load(in_dir+"epoch_{}".format(np.argmin(losses)), map_location=torch.device('cuda'))["model_state_dict"])
model.eval()

# Iterate over each plume
for i in range(28):
    print(i)
    preds = []
    targets = []
    rgb_imgs = []
    diff_imgs = []
        
    test_dataset = MethaneLoader(device = "cuda", mode="val", alli=alli_yn, plume_id=i, channels=channels)

    val_loader = DataLoader(test_dataset, 
                            batch_size = 64, 
                            shuffle = False)


    for batch in val_loader:
        out = model(batch["pred"][:,:,:,:])[...,0].cpu()
        preds.append(out.view(-1,100,100))
        targets.append(batch["target"][...].cpu().view(-1,100,100))
        rgb_imgs.append(batch["rgb_img"][...]*1.5/255)
        diff_imgs.append(batch["diff_img"][...])
    
    if len(preds)>0:
        preds = torch.concat(preds, dim=0).detach().numpy()
        targets = torch.concat(targets, dim=0).detach().numpy()
        rgb_imgs = np.concatenate(rgb_imgs, axis=0)
        diff_imgs = np.concatenate(diff_imgs, axis=0)

        np.save(out_dir+"out_{}.npy".format(i), preds)
        np.save(out_dir+"target_{}.npy".format(i), targets)
        np.save(out_dir+"rgb_{}.npy".format(i), rgb_imgs)
        np.save(out_dir+"diff_{}.npy".format(i), diff_imgs)
import sys
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import torch
from random import shuffle
import random
from numpy import random as npr
import pandas as pd
import imageio
from PIL import Image


class MethaneLoader(DataLoader):
    
    def __init__(self, device, mode, plume_id, red=False, alli=False, channels=12):
        self.device = device
        self.mode = mode
        self.reduce = red
        self.channels = channels

        if mode == "train":
            persist = False
        else:
            persist = True

        if plume_id is not None:
            self.pos_labels = sorted(glob("final_annotations/plume_{}/{}/pos/*.npy".format(plume_id, mode)))
            self.neg_labels = sorted(glob("final_annotations/plume_{}/{}/neg/*.npy".format(plume_id, mode)))
        else:
            self.pos_labels = sorted(glob("final_annotations/plume_*/{}/pos/*.npy".format(mode)))
            self.neg_labels = sorted(glob("final_annotations/plume_*/{}/neg/*.npy".format(mode)))
        
        self.labels = self.pos_labels+self.neg_labels#None
        if not alli:
            self.sample_labels_and_combine(persist=persist)
        
    def sample_labels_and_combine(self, persist=False):
        """
        Sample a subset of negative labels for each epoch
        """
        #if self.mode == "test":
        if self.mode in ["test","val"]:
            self.labels = self.pos_labels+self.neg_labels
        else:
            if persist:
                random.seed(555)
            
            shuffle(self.neg_labels)
            self.labels = self.pos_labels+self.neg_labels[:len(self.pos_labels)]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
    
        f = self.labels[index]
        #print(f)

        plume_id = int(f.split("/")[1].split("_")[1])
        date = f.split("/")[-1][:-4]
        #print(date)

        target = np.load(f)
        context = np.load("turkmenistan_plumes_raw/plume_{}/{}-raw.npy".format(plume_id, date))

        if self.channels==2:
            context = context[...,10:]
        if self.channels==5:
            context = np.concatenate([context[...,1:4], context[...,10:]], axis=-1)

        
        # Crop to centre
        #x_c = target.shape[0]//2
        #y_c = target.shape[1]//2
        s = 50

        if self.mode=="train":
            rng = npr.RandomState()
            mid_loc_x = rng.randint(s,target.shape[0]-s)
            mid_loc_y = rng.randint(s,target.shape[1]-s)

        else:
            mid_loc_x = target.shape[0]//2
            mid_loc_y = target.shape[1]//2

        target = target[mid_loc_x-s:mid_loc_x+s,
                        mid_loc_y-s:mid_loc_y+s]

        context = context[mid_loc_x-s:mid_loc_x+s,
                        mid_loc_y-s:mid_loc_y+s,:]

        diff_img = np.array(diff_img[mid_loc_x-s:mid_loc_x+s,
                                     mid_loc_y-s:mid_loc_y+s,:])
        diff_img_g = np.array(diff_img_g[mid_loc_x-s:mid_loc_x+s,
                                     mid_loc_y-s:mid_loc_y+s])
        rgb_img = np.array(rgb_img[mid_loc_x-s:mid_loc_x+s,
                                     mid_loc_y-s:mid_loc_y+s,:])

        if self.reduce:
            target = np.array([np.int(target.any())])

        #if self.mode == "test":
            #print("Plume ID: {}, date: {}".format(plume_id, date))

        d = {"pred":torch.from_numpy(context).float().to(self.device).permute(2,0,1)/255,
             "target":torch.from_numpy(target).float().to(self.device)}
        
        return d

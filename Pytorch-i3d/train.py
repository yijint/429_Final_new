import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
import numpy as np
import glob
import random
from tensorboardX import SummaryWriter
from preprocess import run_preprocessing

class dataset(torch.utils.data.Dataset):
    
    def __init__(self, paths, v_names, v_labels, num_samples=16): # num_samples cannot be lower than 16
        self.num_samples = num_samples
        self.frames = dict()
        for p in paths:
            self.frames[p] = sorted(glob.glob(p+"/*.jpg"))
        self.data = paths
        self.video_names = v_names
        self.video_labels = v_labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        p = self.data[idx]
        num_frames = len(self.frames[p])-1
        sampled_idx = np.linspace(0, num_frames, self.num_samples) #get num_samples frames from the video
        images = []
        labels = []
        index = video_names.index(p.split('/')[-1]) #index of p's video name in video_names
        labels_video = v_labels[index] #the label for the 
        for i in sampled_idx:
            image = torchvision.io.read_image(self.frames[p][int(i)])
            small_dim = min(image.shape[-2:])
            image = torchvision.transforms.functional.center_crop(image, (small_dim, small_dim))
            image = torchvision.transforms.functional.resize(image, (224, 224), antialias=True)
            images.append(image)
            labels.append(labels_video[i]) # the label for the specific frame
        images = torch.stack(images, axis=1)
        images = (images/255)*2 - 1 #values are between -1 and 1
        return images, labels 

video_names, video_labels = run_preprocessing() #valid names and videos

video_frames_path = "/scratch/network/hishimwe/image" 
# add code here to only extract the videos with v_names and v_labels from preprocess.ipynb 
paths = glob.glob(video_frames_path+"/*")
random.seed(0)
random.shuffle(paths)

good_paths = list(filter(lambda c: c.split('/')[-1] in video_names, paths)) #should only get path where good video name; not sure if this filtering will work 
d=dataset(paths=good_paths, v_names=video_names, v_labels=video_labels)
loader = torch.utils.data.DataLoader(d, shuffle=True, batch_size=10, drop_last=False, num_workers=4)


i3d = InceptionI3d(400, in_channels=3)  #400 is num_classes 
i3d.load_state_dict(torch.load('rgb_imagenet.pt'))
num_classes = len(set(video_labels)) #count unique in labels
i3d.replace_logits(num_classes)
i3d.cuda()

# set up gradient descent params
optimizer = optim.SGD(i3d.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000001) #weight_decay = l2 regularization

lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

# set up training variables 
epochs = 10     #will need to increase later
tot_loss = 0.0
writer = SummaryWriter("deleteme")
step = 0
for e in range(epochs):
    for data, label in loader:
        data = data.cuda()
        label = label.cuda()
        num_frames = data.size(2)
        per_frame_logits = i3d(data).mean(2)
        
        # compute loss 
        loss = F.cross_entropy(per_frame_logits, label) 
        print(f"epoch {e}: loss = {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        writer.add_scalar("train/loss", loss.item(), step) 
        step+=1

writer.flush() # ensure that all loss values are recorded 

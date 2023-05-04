# THINGS TO CHANGE BEFORE RUNNING THIS SCRIPT
# 1. THE CPU YOU ARE USING 
# 2. THE NUMBER OF EPOCHS IN TRAINING
# 3. THE MODEL NAME (FOR SAVING, MAKE SURE YOU HAVE A DIRECTORY NAMED MODELS)

# Import packages
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='3'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<1024>"
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
import time

# Construct a dataset class for training the model
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
        index = np.where(self.video_names == p.split('/')[-1]) #index of p's video name in video_names
        label_video = self.video_labels[index] # the labels for the video
        for i in sampled_idx:
            image = torchvision.io.read_image(self.frames[p][int(i)])
            small_dim = min(image.shape[-2:])
            image = torchvision.transforms.functional.center_crop(image, (small_dim, small_dim))
            image = torchvision.transforms.functional.resize(image, (224, 224), antialias=True)
            images.append(image)
        images = torch.stack(images, axis=1)
        images = (images/255)*2 - 1 #values are between -1 and 1
        return images, label_video 
    
# extract data and labels
video_names, video_labels = run_preprocessing() #valid names and videos
batch_size = 10 # batch size in training
num_videos = len(video_names)

video_frames_path = "/scratch/network/hishimwe/image" 
# only extract the videos with v_names and v_labels from preprocess.ipynb 
paths = glob.glob(video_frames_path+"/*")
random.seed(0)
random.shuffle(paths)

good_paths = list(filter(lambda c: c.split('/')[-1] in video_names, paths)) #should only get path where good video name; not sure if this filtering will work 
d=dataset(paths=good_paths, v_names=video_names, v_labels=video_labels)
loader = torch.utils.data.DataLoader(d, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=4)

# construct the model (should take at most 2 seconds)
i3d = InceptionI3d(400, in_channels=3) # first input is num_classes 
i3d.load_state_dict(torch.load('rgb_imagenet.pt'))
num_classes = len(set(video_labels)) #count unique in labels
i3d.replace_logits(num_classes)
i3d.cuda()

# function to evaluate the model performance
# returns accuracy, f1 score, average f1, and confusion matrix for the data
def eval_metrics(ground_truth, predictions, num_classes):

    #dictionary containing the accuracy, precision, f1, avg f1, and confusion matrix for the data
    f1 = f1_score(y_true=ground_truth, y_pred=predictions, average=None)
    metrics = {
        "accuracy": accuracy_score(y_true=ground_truth, y_pred=predictions),
        "f1": f1,
        "average f1": np.mean(f1),
        "confusion matrix": confusion_matrix(y_true=ground_truth, y_pred=predictions),
        "precision": precision_score(y_true=ground_truth, y_pred=predictions, average=None)
        }
    
    return metrics 

# train
# set up gradient descent params
optimizer = optim.SGD(i3d.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000001) # weight_decay = l2 regularization
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

# set up training variables 
epochs = 2 # will need to increase later
tot_loss = 0.0
writer = SummaryWriter("deleteme")
step = 0
num_batches = np.ceil(num_videos/batch_size)

# train
for e in range(epochs):
    batch_num = 1
    start_time = time.time()
    for data, label in loader:
        data = data.cuda()
        label = label.squeeze().type(torch.LongTensor).cuda()
        num_frames = data.size(2)
        per_frame_logits = i3d(data).mean(2)
        preds = per_frame_logits.cpu().detach().numpy().argmax(axis=1) # convert logits into predictions for evaluating accuracy
        
        # calculate and save loss
        loss = F.cross_entropy(per_frame_logits, label) 
        writer.add_scalar("train/loss", loss.item(), step)
        
        # perform gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        # print metrics every one epoch (at the last batch)
        if (batch_num == num_batches):
            metrics = eval_metrics(ground_truth = label.cpu().detach().numpy(), 
                                   predictions = preds, 
                                   num_classes = num_classes)
            print(f"epoch {e}:")
            print(f"loss: {loss}, accuracy: {round(metrics['accuracy'],2)}, f1 score: {round(metrics['average f1'],2)}")
            print(f"precision: \n{metrics['precision']}")
            print("confusion matrix:")
            print(metrics['confusion matrix']) 
        
        step+=1
        batch_num+=1
        
    print(f"Time taken for epoch {e}: {(time.time()-start_time)/60} mins")
    print("-----------------------------------------------------------------------")

writer.flush() # ensure that all loss values are recorded 

# save model
model_path = "models/baseline_2epochs"
torch.save(i3d, model_path)
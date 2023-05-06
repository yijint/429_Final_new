# TO CHANGE BEFORE RUNNING
is_augment = False
is_dropout = False
dropout_details = "layer1_p0.5"
learning_rate = 0.1
l2 = False
wd = None
lambda1 = 1e-7
num_epochs = 30

# "30epochs_wd_1e-07_dropout__augmented" means the there are 30 training epochs, weight decay is 1e-07, and that there is dropout and augmentation
save_name = f"{num_epochs}epochs"
if (not l2): save_name = save_name + "_l1_lr_" + str(learning_rate) + "_ld_" + str(lambda1) # l1 regularization
if l2: save_name = save_name + "_l2_lr_" + str(learning_rate) + "_wd_"+ str(wd) # l2 regularization
if is_dropout: save_name = save_name + "_dropout_"+dropout_details
if is_augment: save_name = save_name + "_augment"

import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]='3'
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
import numpy as np
from pytorch_i3d import InceptionI3d
import numpy as np
import glob
import random
from tensorboardX import SummaryWriter
from preprocess import run_preprocessing, get_action, holdout_set
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from PIL import Image, ImageSequence

# video augmentation scripts (c) 2018 okankop
from vidaug import *

class dataset(torch.utils.data.Dataset):
    
    def __init__(self, paths, v_names, v_labels, num_samples=16, transforms=None): # num_samples cannot be lower than 16
        self.num_samples = num_samples
        self.frames = dict()
        for p in paths:
            self.frames[p] = sorted(glob.glob(p+"/*.jpg"))
        self.data = paths
        self.video_names = v_names
        self.video_labels = v_labels
        self.transforms = transforms
    
    def __getitem__(self, idx):
        # get original video
        p = self.data[idx]
        
        # sample frames uniformly and create newly sampled video 
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
        
        # data augmentation 
        if (self.transforms is not None):
            images = np.array(self.transforms(images.numpy()))
            # normalize
            images = (images/255)*2 - 1 # values are between -1 and 1
            return torch.from_numpy(images).type(torch.FloatTensor), label_video 
        
        else: 
            images = (images/255)*2 - 1 #values are between -1 and 1
            return images, label_video 

    def __len__(self):
        return len(self.data)
    
if is_augment:
    sometimes = lambda aug: Sometimes(0.4, aug) # Used to apply augmentor with 40% probability
    rand_aug = SomeOf([ # randomly chooses two of the following augmentation methods 
        RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10] 
        RandomTranslate(x=40,y=20), # randomly shifting video in [-x, +x] and [-y, +y] coordinate
        RandomShear(x=0.2,y=0.1), # randomly shearing video in [-x, +x] and [-y, +y] directions.
        sometimes(HorizontalFlip()), # horizontally flip the video with 50% probability
        sometimes(GaussianBlur(sigma=random.uniform(0.5,4))), # blur images using gaussian kernels with std. dev. = sigma
        sometimes(ElasticTransformation(alpha=random.uniform(0,5), cval=int(random.uniform(0,255)), mode="nearest")), # moving pixels locally around using displacement fields
        sometimes(PiecewiseAffineTransform(displacement=15, displacement_kernel=1, displacement_magnification=1)), # places a regular grid of points on an image and randomly moves the neighbourhood of these point around via affine transformations
        sometimes(Add(value=int(random.uniform(-100,100)))), # add a value to all pixel intesities in an video
        sometimes(Multiply(value=2)), # multiply all pixel intensities with given value
        sometimes(Multiply(value=0.5)), # multiply all pixel intensities with given value
        sometimes(Pepper(ratio=25)), # sets a certain fraction of pixel intensities to 0
        sometimes(Salt(ratio=25)), # sets a certain fraction of pixel intensities to 255
    ], 2) # only select two of the above augmenters each time
    
video_train, video_val, label_train, label_val, unique_labels = holdout_set(0.25) #valid names and videos
batch_size = 10 # batch size in training
num_videos_train = len(video_train)
num_videos_val = len(video_val)
num_classes = len(set(label_train)) #count unique in labels

video_frames_path = "/scratch/network/hishimwe/image" 
# only extract the videos with v_names and v_labels from preprocess.ipynb 
paths = glob.glob(video_frames_path+"/*")
random.seed(0)
random.shuffle(paths)

good_paths_train = list(filter(lambda c: c.split('/')[-1] in video_train, paths)) #should only get path where good video name; not sure if this filtering will work 
good_paths_val = list(filter(lambda c: c.split('/')[-1] in video_val, paths)) # validation video paths 

if is_augment: d_train = dataset(paths=good_paths_train, v_names=video_train, v_labels= label_train, transforms=rand_aug)
else: d_train = dataset(paths=good_paths_train, v_names=video_train, v_labels= label_train)
d_val = dataset(paths=good_paths_val, v_names=video_val, v_labels= label_val)

loader_train = torch.utils.data.DataLoader(d_train, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=1)
loader_val = torch.utils.data.DataLoader(d_val, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=1)

start_time = time.time() 
i3d = InceptionI3d(400, in_channels=3) # first input is num_classes in kinetics, this is replaced with replace_logits

if is_dropout: i3d.load_state_dict(torch.load('rgb_imagenet.pt'), strict=False) #added strict = false; theoretically this lets us add layers
else: i3d.load_state_dict(torch.load('rgb_imagenet.pt')) 

i3d.replace_logits(num_classes)
i3d.cuda()

print(f"time taken: {time.time()-start_time} seconds")

#returns accuracy, f1 score, average f1, and confusion matrix for the data
def eval_metrics(ground_truth, predictions, num_classes):

    #dictionary containing the accuracy, precision, f1, avg f1, and confusion matrix for the data
    f1 = f1_score(y_true=ground_truth, y_pred=predictions, labels=np.arange(num_classes), average=None)
    metrics = {
        "accuracy": accuracy_score(y_true=ground_truth, y_pred=predictions),
        "f1": f1,
        "average f1": np.mean(f1),
        "confusion matrix": confusion_matrix(y_true=ground_truth, y_pred=predictions, labels=np.arange(num_classes)),
        "precision": precision_score(y_true=ground_truth, y_pred=predictions, labels=np.arange(num_classes), average=None)
        }
    
    return metrics

def training(model, optimizer, loader, num_classes, reg_type, ld=None):
    losses = []
    ground_truth = []
    predictions = []
    for data, label in loader:
        data = data.cuda()
        label = label.squeeze().type(torch.LongTensor).cuda()
        num_frames = data.size(2)
        per_frame_logits = i3d(data).mean(2)
        preds = per_frame_logits.cpu().detach().numpy().argmax(axis=1) # convert logits into predictions for evaluating accuracy
        
        # calculate and save loss
        loss = F.cross_entropy(per_frame_logits, label)
        losses.append(loss.item()) # append to losses
        ground_truth.extend(list(label.cpu().detach().numpy()))
        predictions.extend(preds.tolist())
        
        if (not reg_type): # l1 regularization
            params = torch.cat([p.view(-1) for p in model.parameters()]) # weights
            norm = torch.norm(params, 1)
            loss = loss - (ld * norm) # updating loss
             
        # back propagation    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    metrics = eval_metrics(ground_truth, predictions, num_classes)   
    return np.mean(losses), metrics # one loss per epoch and the corresponding metrics        
def evaluate(model, loader, num_classes):
    losses = []
    ground_truth = []
    predictions = []
    for data, label in loader:
        data = data.cuda()
        label = label.squeeze().type(torch.LongTensor).cuda()
        num_frames = data.size(2)
        per_frame_logits = i3d(data).mean(2)
        preds = per_frame_logits.cpu().detach().numpy().argmax(axis=1) # convert logits into predictions for evaluating accuracy
        
        # calculate and save loss
        loss = F.cross_entropy(per_frame_logits, label)
        losses.append(loss.item()) # append to losses
        ground_truth.extend(list(label.cpu().detach().numpy()))
        predictions.extend(preds.tolist())
        
    metrics = eval_metrics(ground_truth, predictions, num_classes)
    return np.mean(losses), metrics # one loss per epoch and the corresponding metrics
    
    
# set up gradient descent params

if (l2): # l2 regularization 
    optimizer = optim.SGD(i3d.parameters(), lr=learning_rate, momentum=0.9, weight_decay=wd) # weight_decay = l2 regularization
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
else: # l1 regularization
    optimizer = optim.SGD(i3d.parameters(), lr=learning_rate, momentum=0.9) 
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


# save performance
train_losses = []
train_accuracies = []
train_precisions = []
val_losses = []
val_accuracies = []
val_precisions = []

# train
for e in range(num_epochs):
    start_time = time.time()
    
    print("EPOCH", e)
    
    # training
    loss_train, metrics_train = training(model=i3d, optimizer=optimizer, loader=loader_train, num_classes=num_classes, reg_type=l2, ld=lambda1)
    train_losses.append(loss_train)
    train_accuracies.append(metrics_train["accuracy"])
    train_precisions.append(metrics_train["precision"])
    
    print("TRAINING")
    print("Loss", loss_train)
    print("Accuracy", metrics_train["accuracy"])
    print("Precision", metrics_train["precision"])
    
    # validation 
    loss_val, metrics_val = evaluate(model=i3d, loader=loader_val, num_classes=num_classes)
    val_losses.append(loss_val)
    val_accuracies.append(metrics_val["accuracy"])
    val_precisions.append(metrics_val["precision"])
    
    np.savetxt('/home/jt9744/COS429/429_Final/herve_losses/train/train_'+ save_name, np.array(train_losses), delimiter=",")
    np.savetxt('/home/jt9744/COS429/429_Final/herve_losses/val/val_' + save_name, np.array(val_losses), delimiter=",")

    np.savetxt('/home/jt9744/COS429/429_Final/herve_accuracies/train/train_'+save_name, np.array(train_accuracies), delimiter=",")
    np.savetxt('/home/jt9744/COS429/429_Final/herve_accuracies/val/val_'+save_name, np.array(val_accuracies), delimiter=",")

    np.savetxt('/home/jt9744/COS429/429_Final/herve_precisions/train/train_'+save_name, np.array(train_precisions), delimiter=",")
    np.savetxt('/home/jt9744/COS429/429_Final/herve_precisions/val/val_'+save_name, np.array(val_precisions), delimiter=",")

    print("VALIDATION")
    print("Loss", loss_val)
    print("Accuracy", metrics_val["accuracy"])
    print("Precision", metrics_val["precision"])
        
    print(f"Time taken for epoch {e}: {(time.time()-start_time)/60} mins")
    print("-----------------------------------------------------------------------")
    
print(f"train_losses: {train_losses}")
print(f"val_losses: {val_losses}")
print(f"train_accuracies: {train_accuracies}")
print(f"val_accuracies: {val_accuracies}")

model_path = "/home/jt9744/COS429/429_Final/herve_models_trained/" + save_name 
torch.save(i3d, model_path)

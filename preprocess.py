import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

def run_preprocessing():
    
    train_labels = pd.read_csv('labels/train.txt', delimiter=' ')
    val_labels = pd.read_csv('labels/val.txt', delimiter=' ')
    data_labels = pd.concat([train_labels,val_labels])
    data_labels = data_labels.sort_values(by=['original_vido_id']) # sort alphatebically by video name

    video_names = data_labels["original_vido_id"].to_numpy()
    video_names, unique_idx = np.unique(video_names, return_index = True)
    unique_idx = sorted(unique_idx) # sort the indices so that the video labels order are retained
    labels_str = data_labels["labels"].to_numpy()
    labels_str = labels_str[sorted(unique_idx)]
    labels = []
    for i in range(len(labels_str)):
        this_label = labels_str[i].split(",")
        this_label = [eval(i) for i in this_label]
        labels.append(this_label)
    labels = np.array(labels, dtype=object)

    # find indices of videos with only one class label
    single_class_idx = []
    for i in range(len(labels)):
        if(len(labels[i]) == 1): single_class_idx.append(i)
    # find names and labels of videos with only one class label
    single_class_video_names = video_names[single_class_idx]
    single_class_labels = labels[single_class_idx]
    single_class_labels = [i[0] for i in single_class_labels]
    single_class_labels = np.array(single_class_labels)

    # find the most popular single-class labels 
    c = Counter(single_class_labels)
    top_labels = []
    count_threshold = 100 # count threshold for using a label 
    top_count = 0 # number of data under all top labels with count >= count_threshold
    for i in sorted(c, key=lambda x: -c[x]):
        if (c[i] >= count_threshold): 
            top_labels.append(i)
            top_count += c[i]

    # find indices of single-class videos with enough data (>= count_threshold)
    top_idx = []
    for i in range(len(single_class_labels)):
        if(single_class_labels[i] in top_labels): top_idx.append(i)

    # find names and labels of videos with top single-class labels
    v_names = single_class_video_names[top_idx]
    v_labels = single_class_labels[top_idx]

    # reorder the labels from 0 to num_classes
    video_labels = np.zeros(4805, dtype=np.int8)
    unique_labels = list(set(v_labels))
    for i in range(len(v_labels)):
        video_labels[i] = unique_labels.index(v_labels[i])

    return v_names, video_labels

# function to get action name based on video label 
def get_action(v_label):
    # Get the action name corresponding to each label:
    df_action = pd.read_csv('labels/df_action.txt')
    label_id = df_action["index"].to_numpy()
    action_names = df_action["action"].to_numpy()
    
    return action_names[np.where(label_id==v_label)[0]]
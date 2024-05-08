#!/usr/bin/python3.9
import numpy as np
import os
import torch 
import json
import sys
from utils import Normalizer, multiLSTM_seqformat
from train import train_dataset, train_label, train_ver, normalize, window_size
from deeplog import train_deeplog, test_deeplog_benign, test_deeplog_abnormal, evaluate_roc

# train data
train_dataset = "5g-select"
train_label = "benign"
train_ver = "v5"

model = torch.load(f'save/LSTM_onehot_{train_dataset}_{train_label}_{train_ver}.pth.tar') # trained on 5G-select
print(model)

# test data
test_dataset = "5g-colosseum"
test_label = "abnormal"
test_ver = "v5"

sys.path.append('../../preprocessing/')
from featureV5 import FeatureV5

if "5g-colosseum" in train_dataset or "5g-select" in train_dataset:
    rat = "5G"
else:
    rat = "LTE"

feature = FeatureV5(rat)
num_class = len(feature.keys)

if __name__ == "__main__":
    # load ground truth
    gt = []
    with open(f'../../preprocessing/groundtruth/{test_dataset}_{test_label}_{test_ver}_{window_size}', "r") as i:
        for line in i.readlines():
            tokens = line.strip().split("\t")
            if tokens[2] == "FALSE":
                tokens[2] = False
            elif tokens[2] == "TRUE":
                tokens[2] = True
            gt.append(tokens[2])

    # Validate the performance of trained model
    test_normal_loader = np.load(f'../../preprocessing/data/{train_dataset}_{train_label}_{train_ver}.npz',allow_pickle=True)
    test_normal_seq = test_normal_loader["train_normal_seq"]
    test_normal_label = test_normal_loader["train_normal_label"]

    test_abnormal_loader = np.load(f'../../preprocessing/data/{test_dataset}_{test_label}_{test_ver}.npz',allow_pickle=True)
    test_abnormal_seq = test_abnormal_loader["train_normal_seq"]
    test_abnormal_label = test_abnormal_loader["train_normal_label"]

    # combine 5g-spector with mobile-insight benign for training
    # spector_train_feat = np.load(f'../../preprocessing/data/5g-spector_benign_{train_ver}.npz')
    # spector_train_normal_seq = spector_train_feat['train_normal_seq']
    # spector_train_normal_label = spector_train_feat['train_normal_label']
    # test_normal_seq = np.append(test_normal_seq, spector_train_normal_seq, axis=0)
    # test_normal_label = np.append(test_normal_label, spector_train_normal_label, axis=0)

    # # use the rest 20% for testing
    # random_seed = 42
    # np.random.seed(random_seed)
    # permutation_indices = np.random.permutation(len(test_normal_seq))
    # test_normal_seq = test_normal_seq[permutation_indices]
    # test_normal_label = test_normal_label[permutation_indices]
    # end_index = int(np.floor(len(test_normal_label) * 0.8))
    # test_normal_seq = test_normal_seq[end_index:, :]
    # test_normal_label = test_normal_label[end_index:]
    # test_normal_loader = {}
    # test_normal_loader["train_normal_seq"] = test_normal_seq
    # test_normal_loader["train_normal_label"] = test_normal_label
    # print(test_normal_seq.shape, test_normal_label.shape)

    key_dict = feature.keys

    test_deeplog_benign(model, test_normal_loader, num_class, window_size, key_dict)
    test_deeplog_abnormal(model, test_abnormal_loader, num_class, window_size, key_dict, gt)
    # evaluate_roc(model, test_normal_loader, test_abnormal_loader, num_class, window_size, key_dict, gt)

    # analysis
    analysis = False
    if not analysis:
        exit(0)
    
    
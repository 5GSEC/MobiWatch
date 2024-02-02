#!/usr/bin/python3.9

import pandas as pd

from axolotl.backend.simple import SimpleRunner

from tods.utils import generate_dataset_problem
from tods.search import BruteForceSearch

import numpy as np
import torch
from deeplog import train_deeplog, test_deeplog
from lstm_multivariate import train, test, test_from_iter
from utils import validate_by_rmse, Normalizer
import sys
import more_itertools

# training dataset
train_dataset = "mobileinsight"
train_label = "benign"
train_ver = "v4"
seq_len = 5
normalize = False

def encode_value_within_window(data, is_tmsi=False):
    # encode the values of a vector within a sliding window: e.g., [10, 11, 12] ==> [0, 1, 2]
    new_array = np.array(data)
    unique_idx, indices = np.unique(data, return_inverse=True)
    for unique_id in unique_idx:
        if unique_id == -1 and is_tmsi:
            continue # for TMSI, use -1 to represent no TMSI, so it should not be assigned to 0
        new_array[new_array == unique_id] = np.where(unique_idx == unique_id)[0] / seq_len
    
    return new_array


if __name__ == "__main__":
    # train_feat = np.load('../preprocessing/data/mobileinsight_benign_v1.npy')
    # print(train_feat.shape)
    # normer = Normalizer(train_feat.shape[-1],online_minmax=True)
    # train_feat = normer.fit_transform(train_feat)
    # model, thres = train(train_feat, train_feat.shape[-1])
    # torch.save({'net':model,'thres':thres},'./save/autoencoder_v1.pth.tar')
    
    dataset_name = "%s_%s_%s.npy" % (train_dataset, train_label, train_ver)
    train_feat = np.load('../../preprocessing/data/%s' % (dataset_name))
    print(train_feat.shape)

    if normalize == True:
        normer = Normalizer(train_feat.shape[-1],online_minmax=False)
        train_feat = normer.fit_transform(train_feat)

    X_train = more_itertools.windowed(train_feat,n=seq_len,step=1)
    X_train = np.asarray(list(X_train)[:-1])
    y_train = np.asarray(train_feat[seq_len:])

    # id_column_idx = [0, 1]
    # for x in X_train:
    #     for idx in id_column_idx:
    #         # handle TMSI = -1?
    #         if idx == 1:
    #             is_tmsi = True
    #         else:
    #             is_tmsi = False
    #         encoded_vals = encode_value_within_window(x[:, idx], is_tmsi)
    #         x[:, idx] = encoded_vals

    model, thres = train(train_feat, X_train, y_train)
    torch.save({'net':model,'thres':thres},f'./save/lstm_multivariate_{train_dataset}_{train_label}_{train_ver}.pth.tar')    
    

    # normer = Normalizer(train_feat.shape[-1],online_minmax=True)
    # train_feat = normer.fit_transform(train_feat)
    # model, thres = train(train_feat, train_feat.shape[-1], batch_size, lr, weight_decay, epoches)
    # save_data = {'net':model,'thres':thres, 'dataset':dataset_name, "batch_size":batch_size, "lr":lr, "weight_decay":weight_decay, "epoches":epoches}
    # torch.save(save_data,'./save/autoencoder_%s.pth.tar' % (train_ver))


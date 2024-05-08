#!/usr/bin/python3.9

# Train an autoencoder-based DL model
import numpy as np
import torch
from deeplog import train_deeplog
from utils import validate_by_rmse, Normalizer
import sys

# training dataset
train_dataset = "5g-select"
train_label = "benign"
train_ver = "v5"
window_size = 5
normalize = False

sys.path.append('../../preprocessing/')
from featureV5 import FeatureV5


if __name__ == "__main__":
    
    dataset_name = "%s_%s_%s.npz" % (train_dataset, train_label, train_ver)
    train_feat = np.load('../../preprocessing/data/%s' % (dataset_name))

    train_normal_seq = train_feat['train_normal_seq']
    train_normal_label = train_feat['train_normal_label']
    print(train_normal_seq.shape, train_normal_label.shape)

    # combine 5g-spector with mobile-insight benign for training
    # spector_train_feat = np.load(f'../../preprocessing/data/5g-spector_benign_{train_ver}.npz')
    # spector_train_normal_seq = spector_train_feat['train_normal_seq']
    # spector_train_normal_label = spector_train_feat['train_normal_label']
    # train_normal_seq = np.append(train_normal_seq, spector_train_normal_seq, axis=0)
    # train_normal_label = np.append(train_normal_label, spector_train_normal_label, axis=0)
    # print(train_normal_seq.shape, train_normal_label.shape)

    # # use the first 80% for training
    # random_seed = 42
    # np.random.seed(random_seed)
    # permutation_indices = np.random.permutation(len(train_normal_seq))
    # train_normal_seq = train_normal_seq[permutation_indices]
    # train_normal_label = train_normal_label[permutation_indices]
    # end_index = int(np.floor(len(train_normal_seq) * 0.8))
    # train_normal_seq = train_normal_seq[:end_index, :]
    # train_normal_label = train_normal_label[:end_index]


    if "5g-colosseum" in train_dataset or "5g-select" in train_dataset:
        rat = "5G"
    else:
        rat = "LTE"

    feature = FeatureV5(rat)
    num_class = len(feature.keys)
    print(num_class)
    # encoder = feature.get_one_hot_encoder()

    model = train_deeplog(train_normal_seq, train_normal_label, num_class, window_size)
    torch.save(model, f'./save/LSTM_onehot_{train_dataset}_{train_label}_{train_ver}.pth.tar')


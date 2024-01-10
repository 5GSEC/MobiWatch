#!/usr/bin/python3.9

# Train an autoencoder-based DL model
import numpy as np
import torch
from autoencoder import train, test, test_plot
from utils import validate_by_rmse, Normalizer
import sys

# training dataset
train_dataset = "mobileinsight-all"
train_label = "benign"
train_ver = "v3"

# training parameter
batch_size = 128 
lr = 1e-3 
weight_decay = 1e-6
epoches = 200

# sys.path.append('../../deepaid/')

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
    
    print(train_feat.shape)
    normer = Normalizer(train_feat.shape[-1],online_minmax=True)
    train_feat = normer.fit_transform(train_feat)
    model, thres = train(train_feat, train_feat.shape[-1], batch_size, lr, weight_decay, epoches)
    save_data = {'net':model,'thres':thres, 'dataset':dataset_name, "batch_size":batch_size, "lr":lr, "weight_decay":weight_decay, "epoches":epoches}
    torch.save(save_data,'./save/autoencoder_%s.pth.tar' % (train_ver))


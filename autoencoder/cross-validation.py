#!/usr/bin/python3.9

# Train an autoencoder-based DL model
import numpy as np
import torch
from autoencoder import train, test, test_plot
from utils import validate_by_rmse, Normalizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
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
    dataset_name = "%s_%s_%s.npy" % (train_dataset, train_label, train_ver)
    train_feat = np.load('../../preprocessing/data/%s' % (dataset_name))
    print(train_feat.shape)
    print("Training dataset: %s_%s_%s" % (train_dataset, train_label, train_ver))

    # k-fold cross validation
    k = 5
    kf = KFold(k)
    cnt = 0
    for train_index, test_index in kf.split(train_feat):
        cnt += 1
        print("\n================ K = %d ==================\n" % cnt)
        # print(train_index)
        # training
        X_train, X_test = train_feat[train_index, :], train_feat[test_index, :]
        print(X_train.shape)
        normer = Normalizer(X_train.shape[-1],online_minmax=True)
        X_train = normer.fit_transform(X_train)
        model, thres = train(X_train, X_train.shape[-1], batch_size, lr, weight_decay, epoches, verbose=True)
        # save_data = {'net':model,'thres':thres, 'dataset':dataset_name, "batch_size":batch_size, "lr":lr, "weight_decay":weight_decay, "epoches":epoches}
        # torch.save(save_data,'./save/autoencoder_%s.pth.tar' % (train_ver))

        # testing
        X_test = normer.transform(X_test)
        y_true = 0 * np.ones(X_test.shape[0])
        rmse_vec = test(model,thres,X_test)
        y_pred = np.where(rmse_vec > thres, 1, 0)
        # Calculate accuracy and recall
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        test_dataset = "mobileinsight"
        test_label = "benign"
        test_ver = "v3"
        print("Dataset: %s_%s_%s\tAccuracy: %f" % (test_dataset, test_label, test_ver, accuracy))
        # test_plot(X_test, rmse_vec, thres)

        # load other test datasets
        test_dataset = "phoenix"
        test_label = "abnormal"
        test_ver = "v3"
        test_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (test_dataset, test_label, test_ver))
        test_feat = normer.transform(test_feat)
        y_true = 1 * np.ones(test_feat.shape[0])
        rmse_vec = test(model,thres,test_feat)
        y_pred = np.where(rmse_vec > thres, 1, 0)
        # Calculate accuracy and recall
        accuracy = accuracy_score(y_true, y_pred)
        print("Dataset: %s_%s_%s\tAccuracy: %f" % (test_dataset, test_label, test_ver, accuracy))
        # test_plot(test_feat, rmse_vec, thres)

        test_dataset = "5g-spector"
        test_label = "abnormal"
        test_ver = "v3"
        test_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (test_dataset, test_label, test_ver))
        test_feat = normer.transform(test_feat)
        y_true = 1 * np.ones(test_feat.shape[0])
        rmse_vec = test(model,thres,test_feat)
        y_pred = np.where(rmse_vec > thres, 1, 0)
        # Calculate accuracy and recall
        accuracy = accuracy_score(y_true, y_pred)
        print("Dataset: %s_%s_%s\tAccuracy: %f" % (test_dataset, test_label, test_ver, accuracy))
        # test_plot(test_feat, rmse_vec, thres)

        test_dataset = "5g-spector"
        test_label = "benign"
        test_ver = "v3"
        test_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (test_dataset, test_label, test_ver))
        test_feat = normer.transform(test_feat)
        y_true = 0 * np.ones(test_feat.shape[0])
        rmse_vec = test(model,thres,test_feat)
        y_pred = np.where(rmse_vec > thres, 1, 0)
        # Calculate accuracy and recall
        accuracy = accuracy_score(y_true, y_pred)
        print("Dataset: %s_%s_%s\tAccuracy: %f" % (test_dataset, test_label, test_ver, accuracy))
        # test_plot(test_feat, rmse_vec, thres)

#!/usr/bin/python3.9
import numpy as np
import torch 
import json
from train import train_dataset, train_label, train_ver
from joblib import load

# test data
test_dataset = "mobileinsight"
test_label = "benign"
test_ver = "v3"

# Validate the performance of trained model
test_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (test_dataset, test_label, test_ver))

if __name__ == "__main__":
    # load trained model
    model = load("./save/cblof_%s.joblib" % test_ver)
    print("Model loaded")

    print(test_feat.shape)
    print(test_dataset, test_label, test_ver)
    
    (outlier_labels, confidence) = model.predict(test_feat, return_confidence=True)
    print(confidence)
    
    normal_cnt = test_feat.shape[0] - outlier_labels.__len__()
    abnormal_cnt = outlier_labels.__len__()
    print(normal_cnt, abnormal_cnt)
    if test_label == "abnormal":
        acc = abnormal_cnt / (normal_cnt + abnormal_cnt)
    else:
        acc = normal_cnt / (normal_cnt + abnormal_cnt)
    print("Acc: %f" % acc)


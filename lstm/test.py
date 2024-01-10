#!/usr/bin/python3.9
import numpy as np
import os
import torch 
import json
import sys
from lstm_multivariate import train, test
from utils import Normalizer, multiLSTM_seqformat
from train import train_dataset, train_label, train_ver, normalize, seq_len, encode_value_within_window
from timeseries_multivariate import MultiTimeseriesAID 
import more_itertools  

# train data
train_dataset = "mobileinsight"
train_label = "benign"
train_ver = "v4"

model_dict = torch.load(f'save/lstm_multivariate_{train_dataset}_{train_label}_{train_ver}.pth.tar')
model = model_dict['net']
thres = model_dict['thres']
print(thres)

# test data
test_dataset = "5g-spector"
test_label = "abnormal"
test_ver = "v4"

if __name__ == "__main__":
    print(test_dataset, test_label, test_ver)
    # Validate the performance of trained model
    test_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (test_dataset, test_label, test_ver))
    X_test = more_itertools.windowed(test_feat,n=seq_len,step=1)
    X_test = np.asarray(list(X_test)[:-1])
    y_test = np.asarray(test_feat[seq_len:])

    # id_column_idx = [1, 2]
    # for x in X_test:
    #     for idx in id_column_idx:
    #         # handle TMSI = -1?
    #         if idx == 2:
    #             is_tmsi = True
    #         else:
    #             is_tmsi = False
    #         encoded_vals = encode_value_within_window(x[:, idx], is_tmsi)
    #         x[:, idx] = encoded_vals

    # Load original data for interpretation    
    test_original = np.load(os.path.join('../../preprocessing/data/', "original", f"{test_dataset}_{test_label}_{test_ver}_data_before_encode.npy"))
    X_original = more_itertools.windowed(test_original,n=seq_len,step=1)
    X_original = np.asarray(list(X_original)[:-1])
    y_original = np.asarray(test_original[seq_len:])
    
    # Normalization
    if normalize == True:
        # train data, normalizer
        train_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (train_dataset, train_label, train_ver))
        normer = Normalizer(train_feat.shape[-1],online_minmax=True)
        train_feat = normer.fit_transform(train_feat)
        test_feat = normer.transform(test_feat)
    print(test_feat.shape)

    # Inference
    rmse_vec = test(model,thres,test_feat, X_test, y_test)

    # plot_name = "test_plot_%s_%s_%s" % (test_dataset, test_label, test_ver)
    # test_plot(test_feat, rmse_vec, thres, plot_name)

    print(X_original.__len__(), y_test.__len__())
    i = 110
    print(X_original[i], y_original[i], rmse_vec[i])
    exit(0)
    print(rmse_vec.__len__())

    for i in range(X_original.__len__()):
        r = rmse_vec[i+seq_len-1]
        if r > thres:
            # abnormal
            x = X_original[i]
            print(x)
            y = y_original[i]
            print("==>")
            print(y)
            print()

    normal_cnt = (rmse_vec <= thres).sum()
    abnormal_cnt = (rmse_vec > thres).sum()

    print(normal_cnt, abnormal_cnt)
    # if test_label == "abnormal":
    #     acc = abnormal_cnt / (normal_cnt + abnormal_cnt)
    # else:
    #     acc = normal_cnt / (normal_cnt + abnormal_cnt)
    # print("Acc: %f" % acc)

    # analysis
    analysis = False
    if not analysis:
        exit(0)
    anomaly = test_feat[np.argsort(rmse_vec)[-100]]
    idx = 100
    seq_feat, interp_feat = multiLSTM_seqformat(test_feat, seq_len = seq_len, index=idx)

    """Step 3: Create a DeepAID multivariate Time-Series Interpreter"""
    feature_desc = json.load(open('../../preprocessing/data/desc/%s.json' % test_ver, "r")) # feature_description
    my_interpreter = MultiTimeseriesAID(model,thres,input_size=100,feature_desc=feature_desc)

    """Step 4: Interpret your anomaly and show the result"""
    interpretation = my_interpreter(seq_feat)
    my_interpreter.show_table(interp_feat,interpretation, normer)
    my_interpreter.show_plot(interp_feat, interpretation, normer)
    my_interpreter.show_heatmap(interp_feat,interpretation, normer)
    print(interpretation)
    
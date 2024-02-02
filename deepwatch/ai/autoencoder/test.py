#!/usr/bin/python3.9
import numpy as np
import torch 
import json
from autoencoder import train, test, test_plot
from utils import Normalizer
from train import train_dataset, train_label, train_ver

model_dict = torch.load('save/autoencoder_%s.pth.tar' % train_ver)
model = model_dict['net']
thres = model_dict['thres']
print(thres)

# train data, normalizer
train_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (train_dataset, train_label, train_ver))
normer = Normalizer(train_feat.shape[-1],online_minmax=True)
train_feat = normer.fit_transform(train_feat)

# test data
test_dataset = "phoenix"
test_label = "abnormal"
test_ver = "v3"


if __name__ == "__main__":
    # Validate the performance of trained model
    test_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (test_dataset, test_label, test_ver))
    print(test_dataset, test_label, test_ver)
    test_feat = normer.transform(test_feat)

    print(test_feat.shape)

    rmse_vec = test(model,thres,test_feat)
    plot_name = "test_plot_%s_%s_%s" % (test_dataset, test_label, test_ver)
    test_plot(test_feat, rmse_vec, thres, plot_name)

    normal_cnt = (rmse_vec <= thres).sum()
    abnormal_cnt = (rmse_vec > thres).sum()
    print(normal_cnt, abnormal_cnt)
    if test_label == "abnormal":
        acc = abnormal_cnt / (normal_cnt + abnormal_cnt)
    else:
        acc = normal_cnt / (normal_cnt + abnormal_cnt)
    print("Acc: %f" % acc)

    # analysis
    analysis = True
    if not analysis:
        exit(0)
    import sys
    sys.path.append("./deepaid/interpreters/")
    sys.path.append("./deepaid/")
    from tabular import TabularAID
    feature_desc = json.load(open('../../preprocessing/data/desc/%s.json' % test_ver, "r")) # feature_description
    feature_desc = list(feature_desc.values())
    print(feature_desc.__len__())
    anomaly = test_feat[np.argsort(rmse_vec)[15]]
    my_interpreter = TabularAID(model,thres,input_size=feature_desc.__len__(),feature_desc=feature_desc)

    """Step 4: Interpret your anomaly and show the result"""
    interpretation = my_interpreter(anomaly)
    # DeepAID supports three kinds of visualization of results:
    my_interpreter.show_table(anomaly,interpretation, normer)
    my_interpreter.show_plot(anomaly, interpretation, normer)
    my_interpreter.show_heatmap(anomaly,interpretation, normer)

    # print feature
    for i in range(anomaly.__len__()):
        if anomaly[i] > 0:
            print(feature_desc[i], anomaly[i])
    
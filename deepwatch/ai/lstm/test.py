#!/usr/bin/python3.9
import numpy as np
import os
import torch 
import json
import sys
import pandas as pd
from encoder import Encoder
from lstm_multivariate import train, test
from utils import Normalizer, multiLSTM_seqformat
# from timeseries_multivariate import MultiTimeseriesAID 
import more_itertools  

# train data
train_dataset = "5g-select"
train_label = "benign"

model_dict = torch.load(f'save/lstm_multivariate_{train_dataset}_{train_label}.pth.tar')
model = model_dict['net']
thres = model_dict['thres']
print(thres)

# test data
test_dataset = "5g-colosseum-2"
test_label = "abnormal"

if __name__ == "__main__":
    print(test_dataset, test_label)
    # Validate the performance of trained model
    df = pd.read_csv(f'../autoencoder-mf/data/{test_dataset}_{test_label}_mobiflow.csv', header=0, delimiter=";")
    # Handle missing values
    df.fillna(0, inplace=True)

    sequence_length = 5
    encoder = Encoder()

    X_sequences = encoder.encode_mobiflow(df, sequence_length+1)
    print(X_sequences.shape)

    x_test = []
    y_test = []
    for i in range(len(X_sequences)):
        split_seq = np.split(X_sequences[i], sequence_length+1)
        x_test.append(split_seq[:sequence_length])
        y_test.append(split_seq[-1])

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    print(x_test.shape)
    print(y_test.shape)

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
    # test_original = np.load(os.path.join('../../preprocessing/data/', "original", f"{test_dataset}_{test_label}_{test_ver}_data_before_encode.npy"))
    # X_original = more_itertools.windowed(test_original,n=seq_len,step=1)
    # X_original = np.asarray(list(X_original)[:-1])
    # y_original = np.asarray(test_original[seq_len:])
    
    # # Normalization
    # if normalize == True:
    #     # train data, normalizer
    #     train_feat = np.load('../../preprocessing/data/%s_%s_%s.npy' % (train_dataset, train_label, train_ver))
    #     normer = Normalizer(train_feat.shape[-1],online_minmax=True)
    #     train_feat = normer.fit_transform(train_feat)
    #     test_feat = normer.transform(test_feat)
    # print(test_feat.shape)

    # Inference
    rmse_vec = test(model, thres, x_test, y_test)

    # Convert back to DataFrame
    anomalies = torch.tensor(rmse_vec > thres)
    if len(anomalies) > 0:
        for anomalies_idx in torch.nonzero(anomalies).squeeze():
            df_idx = anomalies_idx
            sequence_data = df.loc[df_idx:df_idx + sequence_length]
            df_sequence = pd.DataFrame(sequence_data, columns=encoder.identifier_features + encoder.numerical_features + encoder.categorical_features)
            print(df_sequence)
            print()
    
    plot = True
    if plot:
        import matplotlib.pyplot as plt
        # Creating a simple line chart
        plt.figure(figsize=(10, 5))
        plt.plot(rmse_vec, marker='o', linestyle='-', color='b')  # Plotting the line chart
        plt.axhline(y=thres, color='r', linestyle='-') # threshold
        plt.title(f'LSTM RMSE (Threshold: {thres})')  # Title of the chart
        plt.xlabel('Seq Index')  # X-axis label
        plt.ylabel('RMSE')  # Y-axis label
        plt.grid(True)  # Adding a grid
        plt.savefig("test.png")  # Display the plot

    # ground truth
    gt = {"blind dos": [10, 21, 32], 
        "downlink dos": [38],
        "downlink imsi extr": [102],
        "uplink imsi extr": list(range(42, 47)),
        "uplink dos": [71, 72],
        "bts ddos": list(range(52, 64))+list(range(88, 97))+list(range(107, 125)),
        "null cipher": list(range(82, 84))
        }

    fn = [v for k in gt.keys() for v in gt[k] ]
    fp = []

    # Convert back to DataFrame
    for anomalies_idx in torch.nonzero(anomalies).squeeze():
        df_idx = anomalies_idx
        sequence_data = df.loc[df_idx:df_idx + sequence_length]
        df_sequence = pd.DataFrame(sequence_data, columns=encoder.identifier_features + encoder.numerical_features + encoder.categorical_features)
        print(df_sequence)

        # evaluation with ground truth
        attack_found = False
        for attack in gt.keys():
            for attack_idx in gt[attack]:
                if anomalies_idx <= attack_idx <= anomalies_idx + sequence_length:
                    attack_found = True
                    if attack_idx in fn:
                        fn.remove(attack_idx) 
            if attack_found:
                break
        
        if attack_found:
            print(f"Attack: {attack}")
        else:
            print(f"False Positive")
            fp.append(anomalies_idx)

        print()

    print("FN:")
    print(fn)
    print("FP:")
    print(fp)

    # plot_name = "test_plot_%s_%s_%s" % (test_dataset, test_label, test_ver)
    # test_plot(test_feat, rmse_vec, thres, plot_name)

    # print(X_original.__len__(), y_test.__len__())
    # i = 110
    # print(X_original[i], y_original[i], rmse_vec[i])
    # exit(0)

    # for i in range(X_original.__len__()):
    #     r = rmse_vec[i+seq_len-1]
    #     if r > thres:
    #         # abnormal
    #         x = X_original[i]
    #         print(x)
    #         y = y_original[i]
    #         print("==>")
    #         print(y)
    #         print()

    # normal_cnt = (rmse_vec <= thres).sum()
    # abnormal_cnt = (rmse_vec > thres).sum()

    # print(normal_cnt, abnormal_cnt)
    # # if test_label == "abnormal":
    # #     acc = abnormal_cnt / (normal_cnt + abnormal_cnt)
    # # else:
    # #     acc = normal_cnt / (normal_cnt + abnormal_cnt)
    # # print("Acc: %f" % acc)

    # # analysis
    # analysis = False
    # if not analysis:
    #     exit(0)
    # anomaly = test_feat[np.argsort(rmse_vec)[-100]]
    # idx = 100
    # seq_feat, interp_feat = multiLSTM_seqformat(test_feat, seq_len = seq_len, index=idx)

    # """Step 3: Create a DeepAID multivariate Time-Series Interpreter"""
    # feature_desc = json.load(open('../../preprocessing/data/desc/%s.json' % test_ver, "r")) # feature_description
    # my_interpreter = MultiTimeseriesAID(model,thres,input_size=100,feature_desc=feature_desc)

    # """Step 4: Interpret your anomaly and show the result"""
    # interpretation = my_interpreter(seq_feat)
    # my_interpreter.show_table(interp_feat,interpretation, normer)
    # my_interpreter.show_plot(interp_feat, interpretation, normer)
    # my_interpreter.show_heatmap(interp_feat,interpretation, normer)
    # print(interpretation)
    
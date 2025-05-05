import numpy as np
import os
import torch 
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
# from timeseries_multivariate import MultiTimeseriesAID 
import more_itertools  
from sklearn.metrics import roc_curve, auc
from .encoder import Encoder
from .lstm_multivariate import train, test
from .utils import Normalizer, multiLSTM_seqformat
from src.mobiflow import UEMobiFlow

target_folder = "./src/ai/lstm_v2"

# Data Preparation
train_dataset = "mobiflow_v2"
train_label = "benign"

test_dataset = "mobiflow_v2"
test_label = "abnormal"

delimeter = ";"

sequence_length = 5

model_dict = torch.load(f"{target_folder}/data/lstm_multivariate_{train_dataset}_{train_label}.pth.tar")
model = model_dict['net']
thres = model_dict['thres']
print(thres)


if __name__ == "__main__":
    print(test_dataset, test_label)
    # Validate the performance of trained model
    data_folder = f"./dataset/{test_dataset}/{test_label}"
    mf_headers = list(UEMobiFlow().__dict__.keys())
    # print(mf_headers)
    csv_files = os.listdir(data_folder)
    encoder = Encoder()
    df_all = pd.DataFrame()
    df_encoded = pd.DataFrame()
    X_sequences = np.array([]) # Create an empty array

    for f in csv_files:
        if f.endswith(".csv"):
            print(f"Processing file: {f}")
            df = pd.read_csv(f'{data_folder}/{f}', delimiter=delimeter, names=mf_headers)
            df = df[df['msg_type'] == 'UE']
            df = df[df['rrc_msg'] != ' ']
            df_all = pd.concat([df_all, df], ignore_index=True)
            df_encoded = pd.concat([df_encoded, encoder.encode(df)], ignore_index=True)

            # encoded_npy = encoded_df.to_numpy()
            # X_sequences = np.vstack([X_sequences, encoded_npy]) if X_sequences.size else encoded_npy

    print(df_encoded.shape)

    # construct sequences
    x_test, y_test = encoder.encode_sequence(df_encoded, sequence_length+1)
    print(x_test.shape, y_test.shape)

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
            sequence_data = df_all.loc[df_idx:df_idx + sequence_length][encoder.get_categorical_features()]
            df_sequence = pd.DataFrame(sequence_data)
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
    # gt = {"blind dos": [10, 21, 32], 
    #     "downlink dos": [38],
    #     "downlink imsi extr": [102],
    #     "uplink imsi extr": list(range(42, 47)),
    #     "uplink dos": [71, 72],
    #     "bts ddos": list(range(52, 64))+list(range(88, 97))+list(range(107, 125)),
    #     "null cipher": list(range(82, 84))
    #     }

    # fn = [v for k in gt.keys() for v in gt[k]]
    # fp = []

    # Convert back to DataFrame
    # for anomalies_idx in torch.nonzero(anomalies).squeeze():
    #     df_idx = anomalies_idx
    #     sequence_data = df_all.loc[df_idx:df_idx + sequence_length][encoder.get_categorical_features()]
    #     df_sequence = pd.DataFrame(sequence_data)
    #     print(df_sequence)
    #     print()

        # evaluation with ground truth
        # attack_found = False
        # for attack in gt.keys():
        #     for attack_idx in gt[attack]:
        #         if anomalies_idx <= attack_idx <= anomalies_idx + sequence_length:
        #             attack_found = True
        #             if attack_idx in fn:
        #                 fn.remove(attack_idx) 
        #     if attack_found:
        #         break
        
        # if attack_found:
        #     print(f"Attack: {attack}")
        # else:
        #     print(f"False Positive")
        #     fp.append(anomalies_idx)

    #     print()

    # print("FN:")
    # print(fn)
    # print("FP:")
    # print(fp)

    # POS = len([v for k in gt.keys() for v in gt[k]])
    # NEG = x_test.shape[0] - POS
    # FP = len(fp)
    # FN = len(fn)
    # TP = POS - FP
    # TN = NEG - FN
    # # Compute precision, recall and F1-measure
    # acc = 100 * (TP + TN) / (TP + TN + FP + FN)
    # P = 100 * TP / (TP + FP)
    # R = 100 * TP / (TP + FN)
    # F1 = 2 * P * R / (P + R)
    # fpr = 100 * FP / (FP + TN)
    # tpr = 100 * TP / (TP + FN)
    # print('false positive (FP): {}, false negative (FN): {}, Acc: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, acc, P, R, F1))
    # print('false positive rate: {:.3f}%, true positive rate: {:.3f}%'.format(fpr, tpr))

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
    
import numpy as np
import pandas as pd
import torch
import sys
import os
import more_itertools
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from .lstm_multivariate import train, test, test_from_iter
from .utils import validate_by_rmse, Normalizer
from .encoder import Encoder
from src.mobiflow import UEMobiFlow

# training dataset
train_dataset = "mobiflow_v2"
train_label = "benign"
delimeter = ";"
target_folder = "./src/ai/lstm_v2"
sequence_length = 5 # use [x0, x1, xn] to predict xn+1

# Step 1: Load and preprocess data
data_folder = f"./dataset/{train_dataset}/{train_label}"
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
x_train, y_train = encoder.encode_sequence(df_encoded, sequence_length+1)
print(x_train.shape, y_train.shape)

seed = 10
val_portion = 0.1 # size of validation set
indices = np.arange(x_train.shape[0])
x_train, x_val, indices_train, indices_val = train_test_split(x_train, indices, test_size=val_portion, random_state=seed)
y_val = y_train[indices_val]
y_train = y_train[indices_train]
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

model, thres, rmse_vec = train(x_train, y_train)
torch.save({'net':model,'thres':thres},f'{target_folder}/data/lstm_multivariate_{train_dataset}_{train_label}.pth.tar')    

# validation
rmse_vec = test(model, thres, x_val, y_val)
anomalies = torch.tensor(rmse_vec > thres)
if len(anomalies) > 0:
    # for anomalies_idx in torch.nonzero(anomalies).squeeze():
    #     df_idx = anomalies_idx
    #     sequence_data = df_all.loc[df_idx:df_idx + sequence_length][encoder.get_categorical_features()]
    #     df_sequence = pd.DataFrame(sequence_data)
    #     print(df_sequence)
    #     print()
    
    FP = len(torch.nonzero(anomalies).squeeze())
    FN = 0
    TP = 0
    TN = x_val.shape[0] - FP
    # Compute precision, recall and F1-measure
    acc = 100 * (TP + TN) / (TP + TN + FP + FN)
    # P = 100 * TP / (TP + FP)
    # R = 100 * TP / (TP + FN)
    # F1 = 2 * P * R / (P + R)
    fpr = 100 * FP / (FP + TN)
    # tpr = 100 * TP / (TP + FN)
    # print('false positive (FP): {}, false negative (FN): {}, Acc: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, acc, P, R, F1))
    print('false positive (FP): {}, false negative (FN): {}, Acc: {:.3f}%'.format(FP, FN, acc))
    print('false positive rate: {:.3f}%'.format(fpr))

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
    plt.savefig(f"{target_folder}/validation.png")  # Display the plot

# normer = Normalizer(train_feat.shape[-1],online_minmax=True)
# train_feat = normer.fit_transform(train_feat)
# model, thres = train(train_feat, train_feat.shape[-1], batch_size, lr, weight_decay, epoches)
# save_data = {'net':model,'thres':thres, 'dataset':dataset_name, "batch_size":batch_size, "lr":lr, "weight_decay":weight_decay, "epoches":epoches}
# torch.save(save_data,'./save/autoencoder_%s.pth.tar' % (train_ver))


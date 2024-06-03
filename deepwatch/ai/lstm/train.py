#!/usr/bin/python3.9

# Train an autoencoder-based DL model
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from lstm_multivariate import train, test, test_from_iter
from utils import validate_by_rmse, Normalizer
from encoder import Encoder
from torch.utils.data import DataLoader, TensorDataset
import sys
import more_itertools

# training dataset
train_dataset = "5g-select"
train_label = "benign"

# Step 1: Load and preprocess data
df = pd.read_csv(f'../autoencoder-mf/data/{train_dataset}_{train_label}_mobiflow.csv', header=0, delimiter=";")
# Handle missing values
df.fillna(0, inplace=True)

sequence_length = 5 # use X to predict next
encoder = Encoder()
X_sequences = encoder.encode_mobiflow(df, sequence_length+1)
print(X_sequences.shape)

x_train = []
y_train = []
for i in range(len(X_sequences)):
    split_seq = np.split(X_sequences[i], sequence_length+1)
    x_train.append(split_seq[:sequence_length])
    y_train.append(split_seq[-1])
    # x_train.append(X_sequences[i][:per_seq_len*(sequence_length)])
    # y_train.append(X_sequences[i][per_seq_len*(sequence_length):])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(x_train.shape)
print(y_train.shape)

# Split data into training and test sets
# seed = 30 # 42
# indices = np.arange(X_sequences.shape[0])
# X_train, X_test, indices_train, indices_test = train_test_split(X_sequences, indices, test_size=0.2, random_state=seed)
# X_train, X_test, indices_train, indices_test = train_test_split(X_sequences, indices, test_size=0.01, random_state=seed)


# Convert to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)

# # Create DataLoader for training
# train_dataset = TensorDataset(X_train, X_train)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# x_train = more_itertools.windowed(X_sequences,n=sequence_length,step=1)
# x_train = np.asarray(list(x_train)[:-1])
# y_train = np.asarray(X_sequences[sequence_length:])
# print(x_train.shape)
# print(y_train.shape)

# id_column_idx = [1, 2]
# for x in X_train:
#     for idx in id_column_idx:
#         # handle TMSI = -1?
#         if idx == 2:
#             is_tmsi = True
#         else:
#             is_tmsi = False
#         encoded_vals = encode_value_within_window(x[:, idx], is_tmsi)
#         x[:, idx] = encoded_vals

model, thres, rmse_vec = train(x_train, y_train)
torch.save({'net':model,'thres':thres},f'./save/lstm_multivariate_{train_dataset}_{train_label}.pth.tar')    

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
    plt.savefig("train.png")  # Display the plot

# normer = Normalizer(train_feat.shape[-1],online_minmax=True)
# train_feat = normer.fit_transform(train_feat)
# model, thres = train(train_feat, train_feat.shape[-1], batch_size, lr, weight_decay, epoches)
# save_data = {'net':model,'thres':thres, 'dataset':dataset_name, "batch_size":batch_size, "lr":lr, "weight_decay":weight_decay, "epoches":epoches}
# torch.save(save_data,'./save/autoencoder_%s.pth.tar' % (train_ver))


import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from .model import Autoencoder
from .encoder import Encoder
from src.mobiflow import UEMobiFlow

target_folder = "./src/ai/autoencoder_v2"

# Data Preparation
train_dataset = "mobiflow_v2"
train_label = "benign"

test_dataset = "mobiflow_v2"
test_label = "abnormal"

delimeter = ";"

# Step 1: Load and preprocess data
data_folder = f"./dataset/{test_dataset}/{test_label}"
mf_headers = list(UEMobiFlow().__dict__.keys())
# print(mf_headers)
csv_files = os.listdir(data_folder)
encoder = Encoder()
df_all = pd.DataFrame()
X_sequences = np.array([]) # Create an empty array

for f in csv_files:
    if f.endswith(".csv"):
        print(f"Processing file: {f}")
        df = pd.read_csv(f'{data_folder}/{f}', delimiter=delimeter, names=mf_headers)
        df = df[df['msg_type'] == 'UE']
        df = df[df['rrc_msg'] != ' ']
        df_all = pd.concat([df_all, df], ignore_index=True)
        encoded_df = encoder.encode(df)
        # print(df)
        # print()
        # print(encoded_df)
        encoded_npy = encoded_df.to_numpy()
        X_sequences = np.vstack([X_sequences, encoded_npy]) if X_sequences.size else encoded_npy

print(X_sequences.shape)

# Convert to PyTorch tensors
X_test = torch.tensor(X_sequences, dtype=torch.float32)

# Load the saved model
input_dim = X_test.shape[1]  # This should match the input_dim used during training
model_path = f"{target_folder}/data/autoencoder_v2_model.pth"

model = Autoencoder(input_dim)
model = torch.load(model_path)['model']
model.eval()
print(f"Model loaded from {model_path}")

# Detect anomalies
with torch.no_grad():
    reconstructions = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructions) ** 2, dim=1)

threshold = torch.load(model_path)['threshold']
anomalies = reconstruction_error > threshold

# Convert back to DataFrame and print anomalies
if len(anomalies) > 0:
    for anomalies_idx in torch.nonzero(anomalies).squeeze():
        abnormal_sequence = df_all.loc[int(anomalies_idx)][encoder.get_categorical_features()]
        # print dataframe in formatted 
        print(abnormal_sequence.to_frame().T)
        print()

print(f"Anomalies detected: {torch.sum(anomalies).item()} out of {len(X_test)} sequences in test set")

# ground truth
# gt = {"blind dos": [10, 21, 32], 
#       "downlink dos": [38],
#       "downlink imsi extr": [102],
#       "uplink imsi extr": list(range(42, 47)),
#       "uplink dos": [71, 72],
#       "bts ddos": list(range(52, 64))+list(range(88, 97))+list(range(107, 125)),
#       "null cipher": list(range(82, 84))
#       }

# fn = [v for k in gt.keys() for v in gt[k] ]
# fp = []

# # Convert back to DataFrame
# for anomalies_idx in torch.nonzero(anomalies).squeeze():
#     df_idx = anomalies_idx
#     sequence_data = df.loc[df_idx:df_idx + sequence_length - 1]
#     df_sequence = pd.DataFrame(sequence_data, columns=encoder.identifier_features + encoder.numerical_features + encoder.categorical_features)
#     print(df_sequence)

#     # evaluation with ground truth
#     attack_found = False
#     for attack in gt.keys():
#         for attack_idx in gt[attack]:
#             if anomalies_idx <= attack_idx < anomalies_idx + sequence_length:
#                 attack_found = True
#                 if attack_idx in fn:
#                     fn.remove(attack_idx) 
#         if attack_found:
#             break
    
#     if attack_found:
#         print(f"Attack: {attack}")
#     else:
#         print(f"False Positive")
#         fp.append(anomalies_idx)

#     print()

# print("FN:")
# print(fn)
# print("FP:")
# print(fp)

# plot graph - reconstruction err w.r.t. to each sequence
plot = True
if plot:
    import matplotlib.pyplot as plt
    # Creating a simple line chart
    plt.figure(figsize=(10, 5))
    plt.plot(reconstruction_error, marker='o', linestyle='-', color='b')  # Plotting the line chart
    plt.axhline(y=threshold, color='r', linestyle='-') # threshold
    plt.title(f'AutoEncoder Reconstruction Error (Threshold: {threshold})')  # Title of the chart
    plt.xlabel('Seq Index')  # X-axis label
    plt.ylabel('AE Error')  # Y-axis label
    plt.grid(True)  # Adding a grid
    plt.savefig(f"{target_folder}/test.png")  # Display the plot


# combine and print anmolous sequences
# indices = torch.nonzero(anomalies).squeeze().tolist()
# combined_ranges = []

# start = end = indices[0]

# for idx in indices[1:]:
#     if idx == end + 1:
#         end = idx
#     else:
#         combined_ranges.append([start, end])
#         start = end = idx

# combined_ranges.append([start, end])
# # print_features = ["rnti", "tmsi", "imsi", "msg", "cipher_alg", "int_alg", "est_cause"]
# print_features = ["rnti", "tmsi", "msg"]
# print("\n\n=====================================\n\n")
# df = df.replace('Securitymodecommand', 'NAS_Securitymodecommand')
# df = df.replace('Securitymodecomplete', 'NAS_Securitymodecomplete')
# df = df.replace('SecurityModeCommand', 'RRC_SecurityModeCommand')
# df = df.replace('SecurityModeComplete', 'RRC_SecurityModeComplete')
# for r in combined_ranges:
#     start = r[0]
#     end = r[1]
#     sequence_data = df.loc[start-sequence_length:end+sequence_length+1]
#     df_sequence = pd.DataFrame(sequence_data, columns=print_features)
#     print(df_sequence.to_string(index=False))
#     print()

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from .model import Autoencoder
from .encoder import Encoder
from src.mobiflow import UEMobiFlow

train_dataset = "mobiflow_v2"
train_label = "benign"
delimeter = ";"
target_folder = "./src/ai/autoencoder_v2"

# Step 1: Load and preprocess data
data_folder = f"./dataset/{train_dataset}/{train_label}"
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

# Split data into training and test sets
seed = 8
indices = np.arange(X_sequences.shape[0])
val_portion = 0.2 # size of validation set
X_train, X_val, indices_train, indices_test = train_test_split(X_sequences, indices, test_size=val_portion, random_state=seed)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# Create DataLoader for training
train_dataset = TensorDataset(X_train, X_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the Autoencoder model
input_dim = X_train.shape[1]
model = Autoencoder(input_dim)

# Compile and train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, _ = data
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Save the model
model_path = f"{target_folder}/data/autoencoder_v2_model.pth"

with torch.no_grad():
    reconstructions = model(X_train)
    reconstruction_error = torch.mean((X_train - reconstructions) ** 2, dim=1)
    percentile = 99
    threshold = np.percentile(reconstruction_error.numpy(), percentile) # we assume the training set contains X% anomalous data
    # alternative: (1) use maximum reconstruction error (2) take average

torch.save({'model': model, "threshold": threshold}, model_path)
print(f"Model saved to {model_path}")


# Detect anomalies
model.eval()

with torch.no_grad():
    reconstructions = model(X_val)
    reconstruction_error = torch.mean((X_val - reconstructions) ** 2, dim=1)

anomalies = reconstruction_error > threshold

# Convert back to DataFrame and print anomalies
if len(anomalies) > 0:
    for anomalies_idx in torch.nonzero(anomalies).squeeze():
        df_idx = indices_test[anomalies_idx]
        abnormal_sequence = df_all.loc[df_idx][encoder.get_categorical_features()]
        print(abnormal_sequence.to_frame().T)
        print()

print(f"Anomalies detected: {torch.sum(anomalies).item()} out of {len(X_val)} sequences in validation set")

# plot graph - reconstruction err w.r.t. to each sequence
plot = True
if plot:
    import matplotlib.pyplot as plt
    # Creating a simple line chart
    plt.figure(figsize=(10, 5))
    plt.plot(reconstruction_error, marker='o', linestyle='-', color='b')  # Plotting the line chart
    plt.axhline(y=threshold, color='r', linestyle='-') # threshold
    plt.title(f'AutoEncoder Reconstruction Error (Threshold: {threshold})')  # Title of the chart
    plt.title('AutoEncoder Reconstruction Error')  # Title of the chart
    plt.xlabel('Seq Index')  # X-axis label
    plt.ylabel('AE Error')  # Y-axis label
    plt.grid(True)  # Adding a grid
    plt.savefig(f"{target_folder}/validation.png")  # Display the plot


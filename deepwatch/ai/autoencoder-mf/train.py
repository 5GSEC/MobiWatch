import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Autoencoder
from encoder import Encoder

train_dataset = "5g-select"
train_label = "benign"

# Step 1: Load and preprocess data
df = pd.read_csv(f'./data/{train_dataset}_{train_label}_mobiflow.csv', header=0, delimiter=";")
# Handle missing values
df.fillna(0, inplace=True)

sequence_length = 6
encoder = Encoder()
X_sequences = encoder.encode_mobiflow(df, sequence_length)
print(X_sequences.shape)

# Split data into training and test sets
seed = 2 # 42
indices = np.arange(X_sequences.shape[0])
val_portion = 0.2 # size of validation set
X_train, X_test, indices_train, indices_test = train_test_split(X_sequences, indices, test_size=val_portion, random_state=seed)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

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
model_path = "./data/autoencoder_model.pth"

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
    reconstructions = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructions) ** 2, dim=1)

anomalies = reconstruction_error > threshold

# Convert back to DataFrame
if len(anomalies) > 0:
    for anomalies_idx in torch.nonzero(anomalies).squeeze():
        df_idx = indices_test[anomalies_idx]
        sequence_data = df.loc[df_idx:df_idx + sequence_length - 1]
        df_sequence = pd.DataFrame(sequence_data, columns=encoder.identifier_features + encoder.numerical_features + encoder.categorical_features)
        print(df_sequence)
        print()

    FP = len(torch.nonzero(anomalies).squeeze())
    FN = 0
    TP = X_test.shape[0] - FP
    TN = 0
    # Compute precision, recall and F1-measure
    acc = 100 * (TP + TN) / (TP + TN + FP + FN)
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    fpr = 100 * FP / (FP + TN)
    tpr = 100 * TP / (TP + FN)
    print('false positive (FP): {}, false negative (FN): {}, Acc: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, acc, P, R, F1))
    print('false positive rate: {:.3f}%, true positive rate: {:.3f}%'.format(fpr, tpr))

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
    plt.savefig("validation.png")  # Display the plot

# Output the anomalies
# anomalous_data = X_test[anomalies]

# # Convert anomalous_data back to original form
# anomalous_data_numpy = anomalous_data.numpy()

# # Reshape anomalous data back to original sequence shape
# num_features_per_line = len(numerical_features) + encoded_identifiers.shape[1] // len(identifier_features) + len(encoder.categories_[0])
# anomalous_data_reshaped = anomalous_data_numpy.reshape(-1, sequence_length, num_features_per_line)

# # Inverse transform numerical features
# anomalous_data_num = scaler.inverse_transform(anomalous_data_reshaped[:, :, :len(numerical_features)].reshape(-1, len(numerical_features)))

# # Inverse transform categorical features
# anomalous_data_cat = encoder.inverse_transform(anomalous_data_reshaped[:, :, len(numerical_features):].reshape(-1, len(encoder.categories_[0])))

# # Combine numerical and categorical features
# anomalous_data_combined = np.hstack([anomalous_data_num, anomalous_data_cat])

# # Convert back to DataFrame
# anomalous_data_list = []
# for i in range(0, anomalous_data_combined.shape[0], sequence_length):
#     sequence_data = anomalous_data_combined[i:i + sequence_length]
#     df_sequence = pd.DataFrame(sequence_data, columns=numerical_features + categorical_features)
#     anomalous_data_list.append(df_sequence)

# print("Anomalous data points in original form:")
# for anomalous_sequence in anomalous_data_list:
#     print(anomalous_sequence)
#     print()

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Autoencoder, positional_encoding
from encoding import nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR


train_dataset = "5g-select"
train_label = "benign"

# Step 1: Load and preprocess data
df = pd.read_csv(f'./data/{train_dataset}_{train_label}_mobiflow.csv', header=0, delimiter=";")
# Handle missing values
df.fillna(0, inplace=True)

# Apply Positional Encoding to identifier features
identifier_features = ['rnti', 'tmsi', 'imsi']
encoded_identifiers = [positional_encoding(df[feature].values, 16) for feature in identifier_features]
encoded_identifiers = np.hstack(encoded_identifiers)

# Encode categorical variables
msg_dicts = [nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR]
possible_categories = {
    'msg': [value for d in msg_dicts for value in d.values()]
}

categorical_features = ['msg']
encoder = OneHotEncoder(categories=[possible_categories[feature] for feature in categorical_features], sparse=False)
encoded_cat_features = encoder.fit_transform(df[categorical_features])

# Normalize numerical variables
# numerical_features = ['ts', 'rnti', 'tmsi', 'imsi', 'imei', 'cipher_alg', 'int_alg', 'est_cause']
numerical_features = ['cipher_alg', 'int_alg']
scaler = StandardScaler()
scaled_num_features = scaler.fit_transform(df[numerical_features])

# Combine features
X = np.hstack([encoded_identifiers, scaled_num_features, encoded_cat_features])

# Reshape data to include sequences of network traces
sequence_length = 5
num_sequences = X.shape[0] - sequence_length + 1
X_sequences = np.array([X[i:i + sequence_length].flatten() for i in range(num_sequences)])

# Split data into training and test sets
indices = np.arange(X_sequences.shape[0])
X_train, X_test, indices_train, indices_test = train_test_split(X_sequences, indices, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoader for training
train_dataset = TensorDataset(X_train, X_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 50  # You can adjust this value based on your requirements
model = Autoencoder(input_dim, encoding_dim)

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
model_path = "./data/autoencoder_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Detect anomalies
model.eval()

with torch.no_grad():
    reconstructions = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructions) ** 2, dim=1)

threshold = np.percentile(reconstruction_error.numpy(), 95)
anomalies = reconstruction_error > threshold


# Convert back to DataFrame
for anomalies_idx in torch.nonzero(anomalies).squeeze():
    df_idx = indices_test[anomalies_idx]
    sequence_data = df.loc[df_idx:df_idx + sequence_length]
    df_sequence = pd.DataFrame(sequence_data, columns=identifier_features + numerical_features + categorical_features)
    print(df_sequence)
    print()


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

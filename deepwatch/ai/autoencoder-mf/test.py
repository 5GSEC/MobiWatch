import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import Autoencoder, positional_encoding
from encoding import nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Data Preparation
train_dataset = "5g-select"
train_label = "benign"

test_dataset = "5g-colosseum-2"
test_label = "abnormal"

# Step 1: Load and preprocess data
df = pd.read_csv(f'./data/{test_dataset}_{test_label}_mobiflow.csv', header=0, delimiter=";")
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

# Convert to PyTorch tensors
X_test = torch.tensor(X_sequences, dtype=torch.float32)

# Load the saved model
input_dim = X_test.shape[1]  # This should match the input_dim used during training
encoding_dim = 50  # This should match the encoding_dim used during training
model_path = "./data/autoencoder_model.pth"

model = Autoencoder(input_dim, encoding_dim)
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Model loaded from {model_path}")

# Detect anomalies
with torch.no_grad():
    reconstructions = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructions) ** 2, dim=1)

threshold = np.percentile(reconstruction_error.numpy(), 75)
anomalies = reconstruction_error > threshold

# Convert back to DataFrame
for anomalies_idx in torch.nonzero(anomalies).squeeze():
    df_idx = anomalies_idx
    sequence_data = df.loc[df_idx:df_idx + sequence_length]
    df_sequence = pd.DataFrame(sequence_data, columns=identifier_features + numerical_features + categorical_features)
    print(df_sequence)
    print()

# Output the anomalies
# anomalous_data = X_test[anomalies]

# # Convert anomalous_data back to original form
# anomalous_data_numpy = anomalous_data.numpy()

# # Reshape anomalous data back to original sequence shape
# num_features_per_line = len(numerical_features) + len(encoder.categories_[0])
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

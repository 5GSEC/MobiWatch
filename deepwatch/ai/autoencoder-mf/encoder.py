import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from encoding import nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR

class Encoder:
    def __init__(self):
        self.identifier_features = ['rnti', 'tmsi']
        self.categorical_features = ['msg']
        self.numerical_features = [] # ['cipher_alg', 'int_alg']
        
        # Categorical variables (msg) encoder
        msg_dicts = [nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR]
        possible_categories = {
            'msg': [value for d in msg_dicts for value in d.values()]
        }
        self.msg_encoder = OneHotEncoder(categories=[possible_categories[feature] for feature in self.categorical_features], sparse_output=False)

        self.id_encoder = None

    # def encode_mobiflow2(self, df: pd.DataFrame, sequence_length: int) -> np.array:

    #     new_df = df.copy()

    #     # Calculate the time difference for 'ts'
    #     new_df['ts'] = df['ts'].diff().fillna(0)
        
    #     # limit ts diff to a threshold
    #     ts_threshold = 10
    #     new_df.loc[abs(new_df['ts']) > ts_threshold, 'ts_diff'] = ts_threshold
        
    #     # Encode 'rnti' as binary feature
    #     new_df['rnti'] = (df['rnti'] == df['rnti'].shift(1)).astype(int).fillna(0)
        
    #     # Encode 'tmsi' as binary feature
    #     new_df['tmsi'] = (df['tmsi'] == df['tmsi'].shift(1)).astype(int).fillna(0)

    #     # Encode categorical variables
    #     msg_dicts = [nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR]
    #     possible_categories = {
    #         'msg': [value for d in msg_dicts for value in d.values()]
    #     }
        
    #     cat_mapping = {value: index+1 for index, value in enumerate(possible_categories['msg'])}
    #     cat_embedding = nn.Embedding(len(possible_categories['msg']), np.floor(np.sqrt(len(possible_categories['msg']))).astype(int))
    #     cat_mapped = df['msg'].map(cat_mapping)
    #     new_df['msg'] = cat_mapped

    #     # Normalize numerical variables
    #     scaler = StandardScaler()
    #     scaled_features = scaler.fit_transform(new_df[['ts', 'rnti', 'tmsi', 'msg']])

    #     # Reshape data to include sequences of network traces
    #     num_sequences = scaled_features.shape[0] - sequence_length + 1
    #     X_sequences = np.array([scaled_features[i:i + sequence_length].flatten() for i in range(num_sequences)])
        
    #     return X_sequences

    def encode_mobiflow(self, df: pd.DataFrame, sequence_length: int) -> np.array:

        # Encode identifier features
        # hash_id_max = 100 # use hash encoding to map sparse ID features into a smaller space
        # id_embedding = nn.Embedding(hash_id_max, np.floor(np.sqrt(hash_id_max)).astype(int))
        
        # # Encode RNTI
        # rnti_tensor = torch.tensor(df['rnti'].values % hash_id_max, dtype=torch.long)
        # encoded_rnti = id_embedding(rnti_tensor)
        # encoded_rnti = encoded_rnti.detach().numpy()

        # # Encode TMSI
        # tmsi_tensor = torch.tensor(df['tmsi'].values % hash_id_max, dtype=torch.long)
        # encoded_tmsi = id_embedding(tmsi_tensor)
        # encoded_tmsi = encoded_tmsi.detach().numpy()

        # # Encode IMSI
        # # imsi_tensor = torch.tensor(df['imsi'].values % hash_id_max, dtype=torch.long)
        # # encoded_imsi = id_embedding(imsi_tensor)
        # # encoded_imsi = encoded_imsi.detach().numpy()

        # # encoded_identifiers = np.hstack([encoded_rnti, encoded_tmsi, encoded_imsi])
        # encoded_identifiers = np.hstack([encoded_rnti, encoded_tmsi])

        # # Encode categorical variables
        # msg_dicts = [nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR]
        # possible_categories = {
        #     'msg': [value for d in msg_dicts for value in d.values()]
        # }

        # encoder = OneHotEncoder(categories=[possible_categories[feature] for feature in self.categorical_features], sparse_output=False)
        # encoded_cat_features = encoder.fit_transform(df[self.categorical_features])
        
        # # cat_mapping = {value: index+1 for index, value in enumerate(possible_categories['msg'])}
        # # cat_dim = np.floor(np.sqrt(len(possible_categories['msg']))).astype(int)
        # # cat_embedding = nn.Embedding(len(possible_categories['msg']), cat_dim)
        # # cat_mapped = df['msg'].map(cat_mapping)
        # # cat_tensor = torch.tensor(cat_mapped.values, dtype=torch.long)
        # # encoded_cat = cat_embedding(cat_tensor)
        # # encoded_cat_features = encoded_cat.detach().numpy()

        # # Normalize numerical variables
        # scaler = StandardScaler()
        # # timestamp
        # ts_diff = df['ts'].diff().fillna(0).to_numpy()
        # # limit ts diff to a threshold
        # ts_threshold = 5
        # ts_diff = np.clip(np.abs(ts_diff), None, ts_threshold)
        # scaled_num_features = scaler.fit_transform(ts_diff.reshape(-1, 1))

        # # scaled_num_features = scaler.fit_transform(df[self.numerical_features])

        # # Combine features
        # # X = np.hstack([encoded_identifiers, scaled_num_features, encoded_cat_features])
        # X = np.hstack([encoded_cat_features])

        # Reshape data to include sequences of network traces
        num_sequences = df.shape[0] - sequence_length + 1
        X_sequences = []
        for i in range(num_sequences):
            seq = df[i:i + sequence_length]
            X_sequences.append(self.encode_sequence(seq, sequence_length))

        # X_sequences = np.array([X[i:i + sequence_length].flatten() for i in range(num_sequences)])

        return np.array(X_sequences)
    
    def encode_sequence(self, df: pd.DataFrame, sequence_len: int) -> np.array:
        encoded_features = []
        # in-sequence encode msg
        if "msg" in self.categorical_features:
            encoded_cat_features = self.msg_encoder.fit_transform(df[self.categorical_features])
            encoded_features.append(encoded_cat_features)

        # in-sequence encode device IDs
        if self.id_encoder is None:
            self.id_encoder = OneHotEncoder(categories=[list(range(0, sequence_len))], sparse_output=False) # max device ID depends on sequence len

        # rnti
        if "rnti" in self.identifier_features:
            unique_rnti = df['rnti'].unique()
            rnti_mapping = {rnti: idx for idx, rnti in enumerate(unique_rnti)}
            rnti_mapped = df['rnti'].map(rnti_mapping)
            encoded_rnti = self.id_encoder.fit_transform(rnti_mapped.values.reshape(-1, 1))
            encoded_features.append(encoded_rnti)

        # tmsi
        if "tmsi" in self.identifier_features:
            unique_tmsi = df['tmsi'].unique()
            tmsi_mapping = {tmsi: idx for idx, tmsi in enumerate(unique_tmsi)}
            tmsi_mapped = df['tmsi'].map(tmsi_mapping)
            encoded_tmsi = self.id_encoder.fit_transform(tmsi_mapped.values.reshape(-1, 1))
            encoded_features.append(encoded_tmsi)

        # imsi
        if "imsi" in self.identifier_features:
            unique_imsi = df['imsi'].unique()
            imsi_mapping = {imsi: idx for idx, imsi in enumerate(unique_imsi)}
            imsi_mapped = df['imsi'].map(imsi_mapping)
            encoded_imsi = self.id_encoder.fit_transform(imsi_mapped.values.reshape(-1, 1))
            encoded_features.append(encoded_imsi)

        X = np.hstack(encoded_features)

        return X.flatten()

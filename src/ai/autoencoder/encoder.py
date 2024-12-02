import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .encoding import nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR

class Encoder:
    def __init__(self):
        self.identifier_features = ['rnti', 's_tmsi']
        self.categorical_features = ['msg']
        self.numerical_features = [] # ['cipher_alg', 'int_alg']
        
        # Categorical variables (msg) encoder
        msg_dicts = [nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR]
        possible_categories = {
            'msg': [value for d in msg_dicts for value in d.values()]
        }
        self.msg_encoder = OneHotEncoder(categories=[possible_categories[feature] for feature in self.categorical_features], sparse_output=False)

        self.id_encoder = None

    def encode_mobiflow(self, df: pd.DataFrame, sequence_length: int) -> np.array:
        # merge rrc_msg and nas_msg into signle feature in MobiFlow v2
        # TODO: develop models of mobiflow v2
        df['msg'] = df['nas_msg'].where(df['nas_msg'] != " ", other=df['rrc_msg'])

        # add rrc setup complete before reg request
        registration_indices = df.index[df['msg'] == 'Registrationrequest'].tolist()
        
        # Duplicate and insert the rows before the matching rows
        for idx in sorted(registration_indices, reverse=True):  # Reverse order to avoid index shifting
            duplicated_row = df.loc[idx].copy()
            duplicated_row["msg"] = "RRCSetupComplete"
            df = pd.concat([df.iloc[:idx], pd.DataFrame([duplicated_row]), df.iloc[idx:]], ignore_index=True)

        # Reshape data to include sequences of network traces
        num_sequences = df.shape[0] - sequence_length + 1
        X_sequences = []
        for i in range(num_sequences):
            if i+sequence_length > df.shape[0]:
                break
            seq = df[i:i + sequence_length]
            X_sequences.append(self.encode_sequence(seq, sequence_length))

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

        # s_tmsi
        if "s_tmsi" in self.identifier_features:
            unique_tmsi = df['s_tmsi'].unique()
            tmsi_mapping = {tmsi: idx+1 for idx, tmsi in enumerate(unique_tmsi)} # tmsi starting from 1
            tmsi_mapping[0] = 0 # 0 tmsi is fixed
            tmsi_mapped = df['s_tmsi'].map(tmsi_mapping)
            encoded_tmsi = self.id_encoder.fit_transform(tmsi_mapped.values.reshape(-1, 1))
            encoded_features.append(encoded_tmsi)

        X = np.hstack(encoded_features)

        return X.flatten()

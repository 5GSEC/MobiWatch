import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from .encoding import nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR

class Encoder:
    def __init__(self):
        self.identifier_features = ['rnti', 's_tmsi']
        self.msg_features = ['rrc_msg', 'nas_msg']
        self.categorical_features = ['rrc_msg', 'nas_msg', 'rrc_cipher_alg', 'rrc_integrity_alg', 'nas_cipher_alg', 'nas_integrity_alg', 'rrc_state', 'nas_state', 'rrc_sec_state']
                                    #  'reserved_field_1', 'reserved_field_2', 'reserved_field_3']
        
        # Categorical variables (msg) encoder
        rrc_msg_dicts = [rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR]
        nas_msg_dicts = [nas_emm_code_NR]
        self.possible_categories = {
            'rrc_msg': [value for d in rrc_msg_dicts for value in d.values()],
            'nas_msg': [value for d in nas_msg_dicts for value in d.values()],
            'rrc_cipher_alg': ['0', '1', '2', '3'], # NEA0, 128-NEA1, 128-NEA2, 128-NEA3...
            'rrc_integrity_alg': ['0', '1', '2', '3'], # NIA0, 128-NIA1, 128-NIA2, 128-NIA3
            'nas_cipher_alg': ['0', '1', '2', '3'], # NEA0, 128-NEA1, 128-NEA2, 128-NEA3
            'nas_integrity_alg': ['0', '1', '2', '3'], # NIA0, 128-NIA1, 128-NIA2, 128-NIA3
            'rrc_state': ['0', '1', '2'],  # e.g., 'RRC_IDLE', 'RRC_INACTIVE', 'RRC_CONNECTED'
            'nas_state': ['0', '1', '2'],  # e.g., 'DEREGISTERED', 'REGISTERED_INITIATED', 'REGISTERED'
            'rrc_sec_state': ['0', '1', '2', '3'],  # e.g., RRC_SEC_CONTEXT_NOT_EXIST, RRC_SEC_CONTEXT_INTEGRITY_PROTECTED, RRC_SEC_CONTEXT_CIPHERED, RRC_SEC_CONTEXT_CIPHERED_AND_INTEGRITY_PROTECTED
        }
        self.possible_categories['nas_msg'].append(" ") # add empty NAS message
        
        # self.msg_encoder = OneHotEncoder(categories=[possible_categories[feature] for feature in self.categorical_features], sparse_output=False)
        # self.id_encoder = None

    def get_categorical_features(self):
        return self.categorical_features
    
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = []
        for feature in self.categorical_features:
            known_values = self.possible_categories[feature]

            # Handle missing values
            df.fillna(0, inplace=True)

            try:
                # Fit the encoder ONLY on the complete list of known values
                onehot_encoder = OneHotEncoder(categories=[known_values], sparse_output=False)
                df_encoded.append(onehot_encoder.fit_transform(df[[feature]]))  # Use double brackets to pass as 2D
            except Exception as e:
                print(f"Error fitting OneHotEncoder for feature '{feature}' with known values: {e}. Skipping.")
                continue

        # Concatenate all encoded features horizontally
        df_encoded = np.hstack(df_encoded)
        
        return pd.DataFrame(df_encoded)


    # def encode_label(self, df: pd.DataFrame) -> pd.DataFrame:
    #     df_encoded = df.copy()

    #     for feature in self.categorical_features:
    #         known_values = self.possible_categories[feature]

    #         # Handle missing values
    #         df.fillna(0, inplace=True)

    #         # Convert the entire column to string type BEFORE transforming
    #         # This ensures consistency with the string values used for fitting
    #         try:
    #             df_encoded[feature] = df_encoded[feature].astype(str)
    #         except Exception as e:
    #             print(f"  Error converting column {feature} to string: {e}. Skipping.")
    #             continue

    #         # Initialize and Fit Encoder ---
    #         le = LabelEncoder()
    #         try:
    #             # Fit the encoder ONLY on the complete list of known values
    #             le.fit(known_values)
    #         except Exception as e:
    #             print(f"Error fitting LabelEncoder for feature '{feature}' with known values: {e}. Skipping.")
    #             continue

    #         # Transform the DataFrame Column ---
    #         try:
    #             # Transform the pre-processed data column
    #             df_encoded[feature] = le.transform(df_encoded[feature])
    #         except ValueError as e:
    #             # This error is CRITICAL. It means a value exists in your DataFrame column
    #             # that was NOT included in your 'all_possible_values[feature]' list.
    #             print(f"  !!! FATAL ERROR transforming feature '{feature}' !!!")
    #             print(f"  Reason: A value encountered in the data was NOT present in the pre-defined list used for fitting.")
    #         except Exception as e:
    #             print(f"  An unexpected error occurred during transform for feature '{feature}': {e}")
        
    #     df_encoded = df_encoded[self.categorical_features]
        
    #     return df_encoded
    

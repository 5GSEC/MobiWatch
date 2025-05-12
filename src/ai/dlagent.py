import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
from io import StringIO
from abc import ABC, abstractmethod
from ..manager import SdlManager
from ..mobiflow import UE_MOBIFLOW_NS, BS_MOBIFLOW_NS, UEMobiFlow, BSMobiFlow
from .formatter import LogFormatter


# import deeplog module
__current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(__current_dir, 'deeplog'))
from .deeplog import MsgSeq
from .deeplog import LSTM_onehot

# import AE module
from .autoencoder import Encoder as AEEncoder
__current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(__current_dir, 'autoencoder'))
from .autoencoder import Autoencoder

# import LSTM module
from .lstm import Encoder as LSTMEncoder
__current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(__current_dir, 'lstm'))
from .lstm import LSTM_multivariate

# import AutoEncoder_v2 module
from .autoencoder_v2 import Encoder as AEEncoder_v2
__current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(__current_dir, 'autoencoder_v2'))
from .autoencoder_v2 import Autoencoder as Autoencoder_v2

# import LSTM_v2 module
from .lstm_v2 import Encoder as LSTMEncoder_v2
__current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(__current_dir, 'lstm_v2'))
from .lstm_v2 import LSTM_multivariate as LSTM_multivariate_v2


class DLAgent(ABC):

    def __init__(self):
        # create console handler with a higher log level
        self.logger = logging.getLogger("mobiwatch")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # avoid double printing
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(LogFormatter())
        self.logger.addHandler(ch)
        # latest mobiflow index read from the database
        self.ue_mobiflow_idx = 0
        self.bs_mobiflow_idx = 0
        self.bs_mobiflow = {}
        self.ue_mobiflow = {}

    def load_mobiflow(self, sdl_mgr: SdlManager) -> dict:
        """
        Load all mobiflow entries from the SDL database

        Parameters:
            sdl_mgr: SDL Manager instance
        """
        start_ue_mf_idx = self.ue_mobiflow_idx
        start_bs_mf_idx = self.bs_mobiflow_idx
        while True:
            resp = sdl_mgr.get_sdl_with_key(UE_MOBIFLOW_NS, self.ue_mobiflow_idx)
            if resp is None or len(resp) == 0:
                break
            else:
                mf_data = resp[str(self.ue_mobiflow_idx)]
                self.ue_mobiflow[self.ue_mobiflow_idx] = mf_data[2:].decode("ascii")  # remove first two non-ascii chars
                self.ue_mobiflow_idx += 1

        while True:
            resp = sdl_mgr.get_sdl_with_key(BS_MOBIFLOW_NS, self.bs_mobiflow_idx)
            if resp is None or len(resp) == 0:
                break
            else:
                mf_data = resp[str(self.bs_mobiflow_idx)]
                self.bs_mobiflow[self.bs_mobiflow_idx] = mf_data[2:].decode("ascii")  # remove first two non-ascii chars
                self.bs_mobiflow_idx += 1
        
        # only return the mobiflow loaded in this query
        if self.ue_mobiflow_idx != 0:
            ret_ue_mf = {key: self.ue_mobiflow[key] for key in range(start_ue_mf_idx, self.ue_mobiflow_idx)}
        else:
            ret_ue_mf = {}
        
        if self.bs_mobiflow_idx != 0:
            ret_bs_mf = {key: self.bs_mobiflow[key] for key in range(start_bs_mf_idx, self.bs_mobiflow_idx)}
        else:
            ret_bs_mf = {}

        return ret_ue_mf, ret_bs_mf 

    @abstractmethod
    def encode(self, raw_data):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

class AutoEncoderAgent_v2(DLAgent):
    def __init__(self, model_path):
        super().__init__()
        self.rat = "5G"
        self.model_path = model_path
        self.model = torch.load(self.model_path)['model']
        self.threshold = torch.load(self.model_path)['threshold']
        self.encoder = AEEncoder_v2()
        logging.info(f"Autoencoder_v2 model loaded, model path: {self.model_path}")
        logging.info(f"{self.model}")
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, ue_mf: dict):
        if ue_mf.__len__() <= 0:
            return None
        # load UE mobiflow keys
        delimiter = ";"
        ue_mf_meta_str = []
        for k in UEMobiFlow().__dict__.keys():
            ue_mf_meta_str.append(k)
        ue_mf_meta_str = delimiter.join(ue_mf_meta_str)

        # construct csv like data from list
        csv_data = ue_mf_meta_str
        for v in ue_mf.values():
            csv_data = csv_data + "\n" + v
        
        df = pd.read_csv(StringIO(csv_data), delimiter=delimiter)

        # df = df[(df["rrc_sec_state"]<1) | (df["rrc_msg"]=="SecurityModeComplete")] # filter messages after encrpytion 
        df.reset_index(drop=True, inplace=True) # reset index

        X_sequences = self.encoder.encode(df)
        X_sequences = X_sequences.to_numpy()

        return X_sequences, df

    def predict(self, seq: np.array) -> list:
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(seq_tensor)
            reconstruction_error = torch.mean((seq_tensor - reconstructions) ** 2, dim=1)
            anomalies = reconstruction_error > self.threshold
            return anomalies.tolist()

    def interpret(self, mf_data: pd.DataFrame, labels: list):
        # Convert back to DataFrame
        for i in range(len(labels)):
            sequence_data = mf_data.loc[i]
            # sequence_data = mf_data.loc[i][self.encoder.get_categorical_features()]
            label = labels[i]
            if label == False:
                self.logger.info(f"\n{sequence_data.to_frame()}")
                self.logger.info("Benign\n\n")
            else:
                self.logger.error(f"\n{sequence_data.to_frame()}")
                self.logger.error("Abnormal\n\n")

class LSTMAgent_v2(DLAgent):
    def __init__(self, model_path, sequence_length=6):
        super().__init__()
        self.rat = "5G"
        self.model_path = model_path
        self.model = torch.load(self.model_path)['net']
        self.threshold = torch.load(self.model_path)['thres']
        self.sequence_length = sequence_length
        self.encoder = LSTMEncoder_v2()
        logging.info(f"LSTM_V2 model loaded, model path: {self.model_path}")
        logging.info(f"{self.model}")
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, ue_mf: dict):
        if ue_mf.__len__() <= 0:
            return None
        # load UE mobiflow keys
        delimiter = ";"
        ue_mf_meta_str = []
        for k in UEMobiFlow().__dict__.keys():
            ue_mf_meta_str.append(k)
        ue_mf_meta_str = delimiter.join(ue_mf_meta_str)

        # construct csv like data from list
        csv_data = ue_mf_meta_str
        for v in ue_mf.values():
            csv_data = csv_data + "\n" + v
        
        df = pd.read_csv(StringIO(csv_data), delimiter=delimiter)

        # df = df[(df["sec_state"]<1) | (df["msg"]=="SecurityModeComplete")] # filter messages after encrpytion 
        df.reset_index(drop=True, inplace=True) # reset index

        # encode features
        df_encoded = self.encoder.encode(df)

        # break the encoded data into sequences
        if len(df) > self.sequence_length:
            X_sequences, Y_sequences = self.encoder.encode_sequence(df_encoded, self.sequence_length)
        else:
            logging.error("Empty data frame, insufficient data")
            X_sequences = []
            Y_sequences = []

        return X_sequences, Y_sequences, df

    def predict(self, x_seq: np.array, y_seq: np.array) -> list:
        # Convert to PyTorch tensors
        x_test = torch.from_numpy(x_seq).type(torch.float).to(self.device)
        y_test = torch.from_numpy(y_seq).type(torch.float).to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(x_test)
            mse_vec = torch.mean((output - y_test) ** 2, dim=1)

            # Convert back to DataFrame
            anomalies = torch.tensor(mse_vec > self.threshold)
            return anomalies.tolist()

    def interpret(self, mf_data: pd.DataFrame, labels: list):
        # Convert back to DataFrame
        for i in range(len(labels)):
            sequence_data = mf_data.loc[i:i + self.sequence_length - 1][self.encoder.get_categorical_features()]
            df_sequence = pd.DataFrame(sequence_data)
            label = labels[i]
            if label == False:
                self.logger.info(f"\n{df_sequence}")
                self.logger.info("Benign\n\n")
            else:
                self.logger.error(f"\n{df_sequence}")
                self.logger.error("Abnormal\n\n")

    def merge_integer_lists(self, input_list_of_lists):
        """
        Merges overlapping lists of integers.

        Takes a list of lists, where each inner list represents a sequence of
        integers. If any two inner lists share one or more common integers,
        they are merged into a single sequence containing all unique integers
        from the merged lists. This process repeats until no more merges are
        possible.

        Args:
          input_list_of_lists: A list of lists of integers.
                               e.g., [[1,2,3], [3,4,5], [7,8,9]]

        Returns:
          A list of lists, where each inner list is a merged sequence
          of unique integers, sorted in ascending order. Returns an empty list
          if the input is empty or contains only empty inner lists.
          e.g., [[1, 2, 3, 4, 5], [7, 8, 9]]
        """
        if not input_list_of_lists:
          return []

        # 1. Convert input lists to a list of sets
        # We filter out any empty lists provided in the input during conversion.
        sets_list = [set(inner_list) for inner_list in input_list_of_lists if inner_list]

        # If after filtering empty lists, the sets_list is empty, return []
        if not sets_list:
            return []

        # 2. Iteratively merge sets that have common elements (same logic as before)
        merged = True  # Flag to check if any merge happened in a pass
        while merged:
          merged = False
          i = 0
          # Iterate through the list of sets
          while i < len(sets_list):
            j = i + 1
            # Compare set 'i' with subsequent sets 'j'
            while j < len(sets_list):
              # Check if sets 'i' and 'j' have any elements in common
              # Using 'isdisjoint()' is efficient for checking overlap
              if not sets_list[i].isdisjoint(sets_list[j]):
                # If they overlap, merge set 'j' into set 'i'
                sets_list[i].update(sets_list[j])
                # Remove the merged set 'j' from the list
                del sets_list[j]
                # A merge occurred, so set the flag to True to repeat the process
                merged = True
                # Do not increment j, because the element at the 'j' index
                # is now the *next* element we need to compare against 'i'.
              else:
                # If no overlap, move to the next set 'j'
                j += 1
            # Move to the next set 'i'
            i += 1

        # 3. Convert the final list of sets into a list of sorted lists
        result_list = [sorted(list(s)) for s in sets_list]

        return result_list


class DeepLogAgent(DLAgent):
    def __init__(self, model_path, window_size=5, ranking_metric="top-k", num_candidates=1, prob_threshold=0.40):
        super().__init__()
        self.rat = "5G"
        self.model_path = model_path
        self.model = torch.load(self.model_path)
        logging.info(f"DeepLog model loaded, model path: {self.model_path}")
        logging.info(f"{self.model}")
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.encoder = MsgSeq()
        self.num_classes = len(self.encoder.get_keys())
        self.ranking_metric = ranking_metric
        self.num_candidates = num_candidates  # top candidates for prediction range
        self.prob_threshold = prob_threshold

    def encode(self):
        """
        Encode a list of MobiFlow records into

        Return:
            x: model input (list of sequence)
            y: model output (list of data)
        """
        x, y = self.encoder.encode(list(self.ue_mobiflow.values()), window_size=self.window_size)
        return x, y

    def encode_mobiflow(self, ue_mf: dict):
        if len(ue_mf.keys()) <= 0:
            return [],[]
        return self.encode(ue_mf)

    def predict(self, seq):
        self.model.eval()
        with torch.no_grad():
            encode_seq = torch.tensor(seq, dtype=torch.long).view(-1, self.window_size).to(self.device)
            encode_seq = F.one_hot(encode_seq, num_classes=self.num_classes).float()
            output = self.model(encode_seq)
            if self.ranking_metric == "top-k":
                ### Use Top candidates
                predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                return predicted
            elif self.ranking_metric == "probability":
                ### Use probability
                probabilities = F.softmax(output, dim=1)
                sorted_probs, indices = torch.sort(probabilities, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=1)
                threshold_idx = (cumulative_probs >= self.prob_threshold).nonzero(as_tuple=True)[1][0]
                predicted = indices[:, :threshold_idx + 1]
                predicted_list = [p.item() for p in predicted[0]]
                return predicted_list
            else:
                raise NotImplementedError
            

    def interpret(self, seq, predicted, actual):
        key_dict = self.encoder.get_keys()
        keys_seq = [key_dict[s] for s in seq]
        keys_predicted = [key_dict[p] for p in predicted]
        keys_actual = key_dict[actual]
        if keys_actual in keys_predicted:
            # Expected, benign outcome
            self.logger.info(f"{keys_seq} => {keys_actual} within prediction {keys_predicted}")
        else:
            # Unexpected, malicious outcome
            self.logger.error(f"{keys_seq} => {keys_actual} out of prediction {keys_predicted}")


class AutoEncoderAgent(DLAgent):
    def __init__(self, model_path, sequence_length=5):
        super().__init__()
        self.rat = "5G"
        self.model_path = model_path
        self.model = torch.load(self.model_path)['model']
        self.threshold = torch.load(self.model_path)['threshold']
        self.sequence_length = sequence_length
        self.encoder = AEEncoder()
        logging.info(f"DeepLog model loaded, model path: {self.model_path}")
        logging.info(f"{self.model}")
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, ue_mf: dict):
        if ue_mf.__len__() <= 0:
            return None
        # load UE mobiflow keys
        delimiter = ";"
        ue_mf_meta_str = []
        for k in UEMobiFlow().__dict__.keys():
            ue_mf_meta_str.append(k)
        ue_mf_meta_str = delimiter.join(ue_mf_meta_str)

        # construct csv like data from list
        csv_data = ue_mf_meta_str
        for v in ue_mf.values():
            csv_data = csv_data + "\n" + v
        
        df = pd.read_csv(StringIO(csv_data), delimiter=delimiter)

        df = df[(df["rrc_sec_state"]<1) | (df["rrc_msg"]=="SecurityModeComplete")] # filter messages after encrpytion 
        df.reset_index(drop=True, inplace=True) # reset index

        if len(df) > self.sequence_length:
            X_sequences = self.encoder.encode_mobiflow(df, self.sequence_length)
        else:
            logging.error("Empty data frame, insufficient data")
            X_sequences = []

        return X_sequences, df

    def predict(self, seq: np.array) -> list:
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(seq_tensor)
            reconstruction_error = torch.mean((seq_tensor - reconstructions) ** 2, dim=1)
            anomalies = reconstruction_error > self.threshold
            return anomalies.tolist()
            

    def interpret(self, mf_data: pd.DataFrame, labels: list):
        # Convert back to DataFrame
        for i in range(len(labels)):
            sequence_data = mf_data.loc[i:i + self.sequence_length - 1]
            df_sequence = pd.DataFrame(sequence_data, columns=self.encoder.identifier_features + self.encoder.numerical_features + self.encoder.categorical_features)
            label = labels[i]
            if label == False:
                self.logger.info(f"\n{df_sequence}")
                self.logger.info("Benign\n\n")
            else:
                self.logger.error(f"\n{df_sequence}")
                self.logger.error("Abnormal\n\n")


class LSTMAgent(DLAgent):
    def __init__(self, model_path, sequence_length=5):
        super().__init__()
        self.rat = "5G"
        self.model_path = model_path
        self.model = torch.load(self.model_path)['net']
        self.threshold = torch.load(self.model_path)['thres']
        self.sequence_length = sequence_length
        self.encoder = LSTMEncoder()
        logging.info(f"DeepLog model loaded, model path: {self.model_path}")
        logging.info(f"{self.model}")
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, ue_mf: dict):
        if ue_mf.__len__() <= 0:
            return None
        # load UE mobiflow keys
        delimiter = ";"
        ue_mf_meta_str = []
        for k in UEMobiFlow().__dict__.keys():
            ue_mf_meta_str.append(k)
        ue_mf_meta_str = delimiter.join(ue_mf_meta_str)

        # construct csv like data from list
        csv_data = ue_mf_meta_str
        for v in ue_mf.values():
            csv_data = csv_data + "\n" + v
        
        df = pd.read_csv(StringIO(csv_data), delimiter=delimiter)

        df = df[(df["sec_state"]<1) | (df["msg"]=="SecurityModeComplete")] # filter messages after encrpytion 
        df.reset_index(drop=True, inplace=True) # reset index

        if len(df) > self.sequence_length:
            X_sequences = self.encoder.encode_mobiflow(df, self.sequence_length)
        else:
            logging.error("Empty data frame, insufficient data")
            X_sequences = []

        return X_sequences, df

    def predict(self, seq: np.array) -> list:
        x_test = []
        y_test = []
        for i in range(len(seq)):
            split_seq = np.split(seq[i], self.sequence_length)
            x_test.append(split_seq[:self.sequence_length-1])
            y_test.append(split_seq[-1])

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        x_test = torch.from_numpy(x_test).type(torch.float).to(self.device)
        y_test = torch.from_numpy(y_test).type(torch.float).to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(x_test)
            mse_vec = torch.mean((output - y_test) ** 2, dim=1)

            # Convert back to DataFrame
            anomalies = torch.tensor(mse_vec > self.threshold)
            return anomalies.tolist()
            

    def interpret(self, mf_data: pd.DataFrame, labels: list):
        # Convert back to DataFrame
        for i in range(len(labels)):
            sequence_data = mf_data.loc[i:i + self.sequence_length - 1]
            df_sequence = pd.DataFrame(sequence_data, columns=self.encoder.identifier_features + self.encoder.numerical_features + self.encoder.categorical_features)
            label = labels[i]
            if label == False:
                self.logger.info(f"\n{df_sequence}")
                self.logger.info("Benign\n\n")
            else:
                self.logger.error(f"\n{df_sequence}")
                self.logger.error("Abnormal\n\n")

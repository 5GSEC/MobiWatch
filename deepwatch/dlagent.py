import torch
import torch.nn.functional as F
import logging
import sys
sys.path.append("./ai/deeplog")

from abc import ABC, abstractmethod

from log_formatter import LogFormatter
from preprocessing.featureV5 import FeatureV5
from ai.deeplog.deeplog import LSTM_onehot

class DLAgent(ABC):

    def __init__(self):
        # create console handler with a higher log level
        self.logger = logging.getLogger("DeepWatch")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False # avoid double printing
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(LogFormatter())
        self.logger.addHandler(ch)

    @abstractmethod
    def predict(self, input_data):
        pass

    @abstractmethod
    def interpret(self, input_data, predicted_data, actual_data):
        pass


class DeepLogAgent(DLAgent):
    def __init__(self, window_size=5, num_candidates=1):
        super().__init__()
        self.train_dataset = "5g-select"
        self.train_label = "benign"
        self.train_ver = "v5"
        self.rat = "5G"
        self.model_path = f"./ai/deeplog/save/LSTM_onehot_{self.train_dataset}_{self.train_label}_{self.train_ver}.pth.tar"
        self.model = torch.load(self.model_path)
        logging.info(f"DeepLog model loaded, model path: {self.model_path}")
        logging.info(f"{self.model}")
        # Model parameters
        feature = FeatureV5(self.rat)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.key_dict = feature.keys
        self.window_size = window_size
        self.num_classes = len(feature.keys)
        self.num_candidates = num_candidates # top candidates for prediction range
    
    # Encode a MobiFlow entry
    # input: mf_data, a list of MobiFlow entry
    # output: window-sliced sequences and labels based on specific window size
    def seq_gen(self, mf_data):
        feature = FeatureV5(self.rat, window_size=self.window_size)
        feature.encode(mf_data, 1)
        return feature.x, feature.y

    def predict(self, seq):
        self.model.eval()
        with torch.no_grad():
            encode_seq = torch.tensor(seq, dtype=torch.long).view(-1, self.window_size).to(self.device)
            encode_seq = F.one_hot(encode_seq, num_classes=self.num_classes).float()
            output = self.model(encode_seq)
            predicted = torch.argsort(output, 1)[0][-self.num_candidates:]        
        return predicted

    def interpret(self, seq, predicted, actual):
        keys_seq = [self.key_dict[s] for s in seq]
        keys_predicted = [self.key_dict[p] for p in predicted]
        keys_actual = self.key_dict[actual]
        if keys_actual in keys_predicted:
            # Expected, benign outcome
            self.logger.info(f"{keys_seq} => {keys_actual} within prediction {keys_predicted}")
        else:
            # Unexpected, malicious outcome   
            self.logger.error(f"{keys_seq} => {keys_actual} out of prediction {keys_predicted}") 


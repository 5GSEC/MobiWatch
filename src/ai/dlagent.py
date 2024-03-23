import torch
import torch.nn.functional as F
import logging
import sys
from abc import ABC, abstractmethod
from ..manager import SdlManager
from ..mobiflow import UE_MOBIFLOW_NS, BS_MOBIFLOW_NS
from .formatter import LogFormatter
from .model.deeplog import MsgSeq


class DLAgent(ABC):

    def __init__(self):
        # create console handler with a higher log level
        self.logger = logging.getLogger("5G-DeepWatch")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # avoid double printing
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(LogFormatter())
        self.logger.addHandler(ch)

    @abstractmethod
    def encode(self, raw_data):
        pass

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
        self.model_path = f"/tmp/src/ai/model/deeplog/save/LSTM_onehot_{self.train_dataset}_{self.train_label}_{self.train_ver}.pth.tar"
        self.model = torch.load(self.model_path)
        logging.info(f"DeepLog model loaded, model path: {self.model_path}")
        logging.info(f"{self.model}")
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.encoder = MsgSeq()
        self.num_classes = len(self.encoder.get_keys())
        self.num_candidates = num_candidates  # top candidates for prediction range

        # latest mobiflow index read from the database
        self.ue_mobiflow_idx = 0
        self.bs_mobiflow_idx = 0
        self.bs_mobiflow = {}
        self.ue_mobiflow = {}

    def load_mobiflow(self, sdl_mgr: SdlManager):
        """
        Load all mobiflow entries from the SDL database

        Parameters:
            sdl_mgr: SDL Manager instance
        """
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

    def encode(self, rw_data):
        """
        Encode a list of MobiFlow records into

        Return:
            x: model input (list of sequence)
            y: model output (list of data)
        """
        x, y = self.encoder.encode(list(self.ue_mobiflow.values()), window_size=self.window_size)
        return x, y

    def encode_mobiflow(self):
        if len(self.ue_mobiflow.keys()) <= 0:
            return None
        self.encode(self.ue_mobiflow)

    def predict(self, seq):
        self.model.eval()
        with torch.no_grad():
            encode_seq = torch.tensor(seq, dtype=torch.long).view(-1, self.window_size).to(self.device)
            encode_seq = F.one_hot(encode_seq, num_classes=self.num_classes).float()
            output = self.model(encode_seq)
            predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
        return predicted

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


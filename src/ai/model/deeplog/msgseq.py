import numpy as np
import more_itertools
from ....mobiflow.mobiflow import UEMobiFlow, MOBIFLOW_DELIMITER
from ....mobiflow.encoding import *

# Treat it as uni-variate time series data, use msg sequence only
class MsgSeq:
    def __init__(self, rat="5G") -> None:
        # load UE mobiflow keys
        ue_mf = UEMobiFlow()
        self.mobiflow_meta_data = []
        for k in ue_mf.__dict__.keys():
            self.mobiflow_meta_data.append(k)

        self.rat = rat
        if rat == "5G":
            list_dicts = [nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR,
                          rrc_ul_dcch_code_NR]
        elif rat == "LTE":
            list_dicts = [nas_emm_code, rrc_dl_ccch_code, rrc_dl_dcch_code, rrc_ul_ccch_code, rrc_ul_dcch_code]
        else:
            raise NotImplementedError
        self.keys = [value for d in list_dicts for value in d.values()]
        # self.keys.append("NULL") # append a new msg to the end to indicate the end
        self.keylen = self.keys.__len__()

        self.selected_feature = ["msg"]
        self.selected_feature_idx = []
        for f in self.selected_feature:
            self.selected_feature_idx.append(self.get_mobiflow_index(f))

    def get_mobiflow_index(self, meta_name):
        return self.mobiflow_meta_data.index(meta_name)

    def get_keys(self):
        return self.keys

    def get_msg_index(self, m: str):
        if m in self.keys:
            return self.keys.index(m)
        else:
            return -1

    def encode(self, trace_list: list, window_size: int):
        """
        Encode a list of MobiFlow record into msg sequences given a specific window size.

        Parameters:
            trace_list (list): list of MobiFlow records
            window_size (int): size of each sequence

        Return:
            x (list): list of sequences for Deeplog inputs
            y (list): list of messages for Deeplog outputs
        """
        mf_data = []
        for trace in trace_list:
            if trace.startswith("BS"):
                continue
            tokens = trace.split(MOBIFLOW_DELIMITER)

            skip = False
            for idx in self.selected_feature_idx:
                feature_name = self.mobiflow_meta_data[idx]
                if feature_name == "msg":
                    msg_idx = self.get_msg_index(tokens[idx])
                    if msg_idx == -1:
                        skip = True
                        break
                    mf_data.append(msg_idx)

            if skip:
                continue

        if len(mf_data) <= 0:
            return

        # data_before_encode.append(len(self.keys)-1) # append a new msg to the end to indicate the end
        data_before_encode = np.array(mf_data)

        # slice into windows
        x = more_itertools.windowed(data_before_encode, n=window_size, step=1)
        x = list(x)[:-1]
        y = list(data_before_encode[window_size:])

        assert len(x) == len(y)

        return x, y


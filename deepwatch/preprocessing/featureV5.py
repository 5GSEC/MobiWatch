from common import *
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from encoding import *
import math
import more_itertools

# Version 5, treat it as uni-variate time series data, use msg sequence only
class FeatureV5:
    def __init__(self, rat="5G") -> None:
    
        self.mobiflow_meta_str = "msg_type;msg_id;ts;ver;gen;bs_id;rnti;tmsi;imsi;imei;cipher_alg;int_alg;est_cause;msg;rrc_state;nas_state;sec_state;emm_cause;rrc_init_timer;rrc_inactive_timer;nas_initial_timer;nas_inactive_timer"
        self.delimeter = ";"
        self.mobiflow_meta_data = self.mobiflow_meta_str.split(self.delimeter)
        self.rat = rat

        if rat == "5G":
            list_dicts = [nas_emm_code_NR, rrc_dl_ccch_code_NR, rrc_dl_dcch_code_NR, rrc_ul_ccch_code_NR, rrc_ul_dcch_code_NR]
        elif rat == "LTE":
            list_dicts = [nas_emm_code, rrc_dl_ccch_code, rrc_dl_dcch_code, rrc_ul_ccch_code, rrc_ul_dcch_code]
        else:
            raise NotImplementedError
        self.keys = [value for d in list_dicts for value in d.values()]
        self.keys.append("NULL") # append a new msg to the end to indicate the end
        self.keylen = self.keys.__len__()

        self.selected_feature = ["msg"]
        self.selected_feature_idx = []
        for f in self.selected_feature:
            self.selected_feature_idx.append(self.get_mobiflow_index(f))

        # time series parameters
        self.window_size = 5

        self.x = []
        self.y = []
        self.x_before_encode = []
        self.y_before_encode = []
        self.labels = []
        self.desc = None
        self.data_before_encode = []

    def select_feature_length(self):
        return self.selected_feature.__len__()
    
    def get_mobiflow_index(self, meta_name):
        return self.mobiflow_meta_data.index(meta_name)
    
    def get_msg_index(self, m):
        if m in self.keys:
            return self.keys.index(m)
        else:
            return -1
        
    def normalize_msg_idx(self, idx):
        return idx / (self.keys.__len__()-1)
    
    def get_msg_by_index(self, idx):
        return self.keys[idx]
    
    def get_feature_description(self):
        return self.feature_desc
    
    def get_one_hot_encoder(self):
        return OneHotEncoder(categories=[self.keys], sparse_output=False, handle_unknown='ignore') # we know the total number of msg types, so specify it

    # Input: mobiflow trace list of a single file
    def encode(self, trace_list, label):
        data_before_encode = []
        for trace in trace_list:
            if trace.startswith("BS"):
                continue
            tokens = trace.split(self.delimeter)

            arr = []
            skip = False
            for idx in self.selected_feature_idx:
                feature_name = self.mobiflow_meta_data[idx]
                if feature_name == "msg":
                    msg_idx = self.get_msg_index(tokens[idx])
                    if msg_idx == -1:
                        skip = True
                        break
                    data_before_encode.append(msg_idx)
            
            if skip:
                continue
        
        if len(data_before_encode) <= 0:
            return
        
        data_before_encode.append(len(self.keys)-1) # append a new msg to the end to indicate the end
        data_before_encode = np.array(data_before_encode)
         
        # one-hot encoding
        # encoder = self.get_one_hot_encoder()

        # data_after_encode = encoder.fit_transform(data_before_encode)
        # self.desc = encoder.categories_
        # print(data_after_encode.shape)

        # slice into windows
        x = more_itertools.windowed(data_before_encode,n=self.window_size,step=1)
        x = list(x)[:-1]
        y = list(data_before_encode[self.window_size:])

        # convert into deeplog dict format
        self.x = self.x + x
        self.y = self.y + y

        self.labels = self.labels + ([label] * len(x))

        print(len(self.x))


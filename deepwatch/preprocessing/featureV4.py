from common import *
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from encoding import *
import math

# Version 4, treat it as multi-variate time series data
class FeatureV4:
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
        self.keylen = self.keys.__len__()

        self.rnti_dict = {}
        self.tmsi_dict = {}

        self.selected_feature = []
        # self.feature_desc = ["msg", "rnti", "tmsi"]
        self.feature_desc = ["msg"]
        for f in self.feature_desc:
            self.selected_feature.append(self.get_mobiflow_index(f))

        self.data = None
        self.data_before_encode = None

    def rnti_encode(self, rnti):
        if not rnti in self.rnti_dict.keys():
            self.rnti_dict[rnti] = self.rnti_dict.__len__()
        return self.rnti_dict[rnti]
    
    def tmsi_encode(self, tmsi):
        if tmsi == 0:
            return -1
        if not tmsi in self.tmsi_dict.keys():
            self.tmsi_dict[tmsi] = self.tmsi_dict.__len__()
        return self.tmsi_dict[tmsi]

    def rnti_encode_cos(self, rnti):
        RNTI_MAX = 0xffff
        d = 4
        res = []
        for i in range(d):
            if i % 2 == 0:
                res.append(math.sin(rnti/pow(RNTI_MAX, 2*i/d)))
            else:
                res.append(math.cos(rnti/pow(RNTI_MAX, 2*i/d)))
        return res
    
    def tmsi_encode_cos(self, tmsi):
        TMSI_MAX = 10000 # TMSI max value should be 2^32, use 10000 for now...
        d = 4
        res = []
        for i in range(d):
            if i % 2 == 0:
                res.append(math.sin(tmsi/pow(TMSI_MAX, 2*i/d)))
            else:
                res.append(math.cos(tmsi/pow(TMSI_MAX, 2*i/d)))
        return res

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
    
    # def get_transformer(self):
    #     transformer = []
    #     for f in self.feature_desc:
    #         idx = self.feature_desc.index(f)
    #         if f == "msg":
    #             # use one-hot encoder
    #             one_hot_encoder = OneHotEncoder(categories=[self.keys], sparse_output=False, handle_unknown='ignore') # we know the total number of msg types, so specify it
    #             transformer.append(('onehot', one_hot_encoder, [idx]))
    #         elif f == "rnti" or f == "tmsi":
    #             # use ordinal encoder for identifiers
    #             found = False
    #             for t in transformer:
    #                 if t[0] == 'ord':
    #                     t[2].append(idx) # ordinal encoder exist, just append index to it
    #                     found = True
    #                     break
    #             if not found:
    #                 transformer.append(('ord', OrdinalEncoder(), [idx])) # create ordinal encoder
    #         else:
    #             pass # rest features
                    
    #     transformer = ColumnTransformer(
    #         transformers=transformer,
    #         remainder='passthrough'
    #     )
    #     return transformer

    def get_transformer(self):
        transformer = []
        for f in self.feature_desc:
            idx = self.feature_desc.index(f)
            if f == "msg":
                # use one-hot encoder
                found = False
                for t in transformer:
                    if t[0] == "onehot":
                        t[2].append(idx)
                        found = True
                        break
                if not found:
                    max_cat = self.keys.__len__()
                    one_hot_encoder = OneHotEncoder(categories=[self.keys], sparse_output=False, handle_unknown='ignore') # we know the total number of msg types, so specify it
                    transformer.append(('onehot', one_hot_encoder, [idx]))
            else:
                pass # rest features
                    
        transformer = ColumnTransformer(
            transformers=transformer,
            remainder='passthrough'
        )
        return transformer

    def encode(self, trace_list, label):
        # self.data = np.empty(shape=(0, self.selected_feature.__len__()))
        self.data = []
        for trace in trace_list:
            if trace.startswith("BS"):
                continue
            tokens = trace.split(self.delimeter)

            arr = []
            skip = False
            for idx in self.selected_feature:
                feature_name = self.mobiflow_meta_data[idx]
                if feature_name == "rnti":
                    arr.append(self.rnti_encode(int(tokens[idx])))
                    # arr = arr + self.rnti_encode_cos(int(tokens[idx])) # use sin-cos encoding
                elif feature_name == "tmsi":
                    arr.append(self.tmsi_encode(int(tokens[idx])))
                    # arr = arr + self.tmsi_encode_cos(int(tokens[idx])) # use sin-cos encoding
                # elif feature_name == "msg":
                #     idx = self.get_msg_index(tokens[idx])
                #     if idx == -1:
                #         skip = True
                #         break
                #     arr.append(self.normalize_msg_idx(idx))
                else:
                    arr.append(tokens[idx])
            
            if skip:
                continue
            arr = np.array(arr)
            self.data.append(arr)
        
        self.data = np.array(self.data)
        
        # one-hot encoding
        transformer = self.get_transformer()

        print(self.data.shape)
        self.data_before_encode = self.data.copy()
        self.data = transformer.fit_transform(self.data)
        self.data = np.array(self.data, dtype=float)
        print(self.data.shape)

        labels = np.zeros(self.data.shape[0]) * label


        return self.data, labels, self.get_feature_description()

        

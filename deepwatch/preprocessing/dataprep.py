#!/usr/bin/python3.9
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from feature import Feature
from featureV2 import FeatureV2
from featureV3 import FeatureV3
from featureV4 import FeatureV4
from featureV5 import FeatureV5


class DataLoader:
    def __init__(self):
        self.dataset_phoenix = "/home/wen.423/Desktop/5g/dataset/phoenix"
        self.dataset_5g_spector = "/home/wen.423/Desktop/5g/dataset/5g-spector"
        self.dataset_mobileinsight = "/home/wen.423/Desktop/5g/mobileinsight-core/examples/jsonlogs"
        self.dataset_5g_colosseum = "/home/wen.423/Desktop/5g/5g-ai/colosseum-logs/1024-NR-5UE-10011-NORMAL/osu-seclab-oai-secsm-ran-ue-img-1021-srn59-RES139156/pcaps"
        
        self.mobiflow_phoenix = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/phoenix"
        self.mobiflow_5g_spector = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/5g-spector"
        self.mobiflow_mobileinsight = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/mobile-insight"
        self.mobiflow_5g_colosseum = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/5g-colosseum"

        self.dataset_name = ""
        self.abnormal = None
        self.ver = 0
        # self.datasets = [self.dataset_phoenix, self.dataset_mobileinsight, self.dataset_phoenix]
        # self.mobiflow = [self.mobiflow_5g_spector, self.mobiflow_phoenix, self.mobiflow_mobileinsight]

        self.mobiflow_meta_str = "msg_type;msg_id;ts;ver;gen;bs_id;rnti;tmsi;imsi;imei;cipher_alg;int_alg;est_cause;msg;rrc_state;nas_state;sec_state;emm_cause;rrc_init_timer;rrc_inactive_timer;nas_initial_timer;nas_inactive_timer"
        self.delimeter = ";"
        self.mobiflow_meta_data = self.mobiflow_meta_str.split(self.delimeter)

        self.trace_list = []
        self.data = None
        self.labels = None
        self.feature_description = {}

        self.data_out_path = os.path.join(os.getcwd(), "data")


    def _reset(self):
        self.dataset_name = ""
        self.abnormal = None
        self.ver = 0
        self.data = None
        self.labels = None
        self.trace_list = []

    def load_data(self, dataset_name, abnormal, ver):
        self._reset()
        self.dataset_name = dataset_name
        self.abnormal = "abnormal" if abnormal == True else "benign"
        self.ver = ver
        if self.dataset_name.__contains__("5g-colosseum"):
            self.rat = "5G"
        else:
            self.rat = "LTE"

        if dataset_name == "phoenix":
            mf_folder = self.mobiflow_phoenix
            if abnormal == True:
                forbidden_set = []
                type = "abnormal"
                label = 1
            else:
                raise NotImplementedError
            
        elif dataset_name == "5g-spector":
            mf_folder = self.mobiflow_5g_spector
            normal_set = ["normal_du.txt"]
            if abnormal == True:
                # forbidden_set = normal_set + ["blind_dos_ue.txt", "bts_resource_depletion_ota_ue.txt", "bts_resource_depletion_ue.txt"]
                forbidden_set = normal_set
                type = "abnormal"
                label = 1
            else:
                forbidden_set = list(set(os.listdir(mf_folder)) ^ set(normal_set))
                type = "benign"
                label = 0

        elif dataset_name == "mobileinsight":
            mf_folder = self.mobiflow_mobileinsight
            ab_set = ["diag_log_20150727_200409_Samsung-SM-G900T_ATT.txt", "diag_log_20150729_085300_LGE-LGLS660_Sprint.txt", "diag_log_20150726_065902_LGE-LGLS660_Sprint.txt",
                             "diag_log_20150726_164823_LGE-LGLS660_Sprint.txt", "diag_log_20150727_200409_Samsung-SM-G900T_ATT.txt", "diag_log_20150727_203911_LGE-LGLS660_Sprint.txt",
                             "two-default-bearers-verizion-volte.txt"]
            if abnormal == True:
                type = "abnormal"
                forbidden_set = list(set(os.listdir(mf_folder)) ^  set(ab_set))
                label = 1
            else:
                type = "benign"
                forbidden_set = ab_set
                label = 0
            
        elif dataset_name == "mobileinsight-all":
            # in this set we don't separate benign and abnormal
            mf_folder = self.mobiflow_mobileinsight
            type = "benign"
            forbidden_set = []
            label = 0

        elif dataset_name == "5g-colosseum":
            mf_folder = self.mobiflow_5g_colosseum
            if abnormal == True:
                mf_folder = os.path.join(mf_folder, "attack")
                type = "abnormal"
                forbidden_set = []
                label = 1
            else:
                mf_folder = os.path.join(mf_folder, "benign")
                type = "benign"
                forbidden_set = []
                label = 0
        else:
            raise NotImplementedError

        # print(self.data) 
        print("Dataset: %s, Abnormal: %s" % (dataset_name, str(abnormal)))
        if ver <= 3:
            self.data, self.labels, self.feature_description = self.construct_feature(mf_folder, forbidden_set, label, ver)
        elif ver == 4 or ver == 5:
            self.data, self.labels, self.feature_description = self.construct_time_series(mf_folder, forbidden_set, label, ver)
        # print(self.data.shape, self.labels.shape)

        # Save data
        if ver == 5:
            out_name = "%s_%s_v%d.npz" % (dataset_name, type, ver) # save as npz file for deeplog data
            np.savez("%s/%s" % (self.data_out_path, out_name), train_normal_seq=self.data["train_normal_seq"], train_normal_label=self.data["train_normal_label"])
        else:
            out_name = "%s_%s_v%d.npy" % (dataset_name, type, ver)
            np.save("%s/%s" % (self.data_out_path, out_name), self.data)
        
            # Save feature description
            with open(os.path.join(self.data_out_path, "desc", "v%s.json" % ver), "w") as f:
                f.write(json.dumps(self.feature_description, indent=4))


    def get_mobiflow_index(self, meta_name):
        return self.mobiflow_meta_data.index(meta_name)
    
    def corr(self):
        # cor_x = self.data[:, 0:-1]
        # cor_y = np.matrix(self.data[:, -1])
        # print(cor_x.shape, cor_y.shape)
        # cor = np.corrcoef(cor_x, cor_y)
        # print(cor)

        df = pd.DataFrame(self.data)
        cor = df.corr()
        print(cor)
        threshold = 0.3
        idx = np.argwhere((np.abs(cor) > threshold) & (np.abs(cor) < 1))
        f = Feature()
        for (x, y) in idx:
            print(list(f.feature.keys())[x], list(f.feature.keys())[y], cor[x][y])
    

    # batch of files
    def construct_feature(self, folder, forbidden_set, label, ver):
        # Init feature vec
        if ver == 1:
            feature = Feature()
        elif ver == 2:
            feature = FeatureV2()
        elif ver == 3:
            feature = FeatureV3()
        else:
            raise NotImplementedError
        
        fs = os.listdir(folder)
        for f in fs:
            if f in forbidden_set:
                continue
            f = os.path.join(folder, f)
            print(f)
            if ver <= 3:
                self._construct_feature(f, label, ver)
            else:
                raise NotImplementedError

        return feature.encode(self.trace_list, label)

    # single file
    def _construct_feature(self, file_name, label, ver):
        new_feature_flag = False
        # session-based or time-based split
        start_ts = 0
        ts_threshold = 20000 # in ms
        msg_list = []
        last_rnti = None
        with open(file_name, "r") as i:
            for line in i.readlines():
                tokens = line.split(";")
                rnti = tokens[self.get_mobiflow_index("rnti")]
                ts = float(tokens[self.get_mobiflow_index("ts")])
                msg = tokens[self.get_mobiflow_index("msg")]
                # session based split using RNTI
                if last_rnti is not None and last_rnti != rnti:
                    # new feature vector
                    new_feature_flag = True
                elif last_rnti is None:
                    new_feature_flag = False
                
                last_rnti = rnti

                if new_feature_flag:
                    # record old feature
                    self.trace_list.append(msg_list)

                    # start a new feature vector
                    msg_list = []
                    new_feature_flag = False

                # add data
                msg_list.append(msg)
            
            if msg_list != []:
                self.trace_list.append(msg_list)

    # batch of files
    def construct_time_series(self, folder, forbidden_set, label, ver):
        ### Multi-variate Time series
        if ver == 4:
            feature = FeatureV4(self.rat)
            fs = os.listdir(folder)
            trace_list = []

            for f in fs:
                if f in forbidden_set:
                    continue
                f = os.path.join(folder, f)
                print(f)
                with open(f, "r") as i:
                    trace_list = trace_list + i.readlines()
            
            # construct data
            data, labels, desc = feature.encode(trace_list, label)

            # save data before encoding
            data_path = os.path.join(self.data_out_path, "original", f"{self.dataset_name}_{self.abnormal}_v{self.ver}_data_before_encode.npy")
            np.save(data_path, feature.data_before_encode)
            
            # save msg mapping
            with open(os.path.join(self.data_out_path, "desc", "v%s_msg_encoding.json" % ver), "w") as f:
                f.write(json.dumps(feature.keys, indent=4))
            return data, labels, desc
        ### Uni-variate Time series
        elif ver == 5:
            feature = FeatureV5(self.rat)
            fs = os.listdir(folder)
            for f in fs:
                trace_list = []
                if f in forbidden_set:
                    continue
                f = os.path.join(folder, f)
                print(f)
                with open(f, "r") as i:
                    trace_list = trace_list + i.readlines()
                feature.encode(trace_list, label)
            
            data = {}
            data["train_normal_seq"] = feature.x
            data["train_normal_label"] = feature.y
            return data, feature.labels, feature.desc
        else:
            raise NotImplementedError
        


ver = 5
dl = DataLoader()
dl.load_data("mobileinsight", False, ver)
dl.load_data("mobileinsight", True, ver)
dl.load_data("phoenix", True, ver)
dl.load_data("5g-spector", True, ver)
dl.load_data("5g-spector", False, ver)
dl.load_data("mobileinsight-all", False, ver)
dl.load_data("5g-colosseum", True, ver)
dl.load_data("5g-colosseum", False, ver)


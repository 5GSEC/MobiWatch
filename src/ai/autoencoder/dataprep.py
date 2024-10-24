#!/usr/bin/python3.9
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# pcap ==> mobiflow
class DataLoader:
    def __init__(self):
        self.dataset_5g_benign = "../../dataset/phoenix"
        self.dataset_5g_attack = "/home/wen.423/Desktop/5g/dataset/phoenix"

        self.dataset_phoenix = "/home/wen.423/Desktop/5g/dataset/phoenix"
        self.dataset_phoenix = "/home/wen.423/Desktop/5g/dataset/phoenix"
        self.dataset_5g_spector = "/home/wen.423/Desktop/5g/dataset/5g-spector"
        self.dataset_mobileinsight = "/home/wen.423/Desktop/5g/mobileinsight-core/examples/jsonlogs"
        self.dataset_5g_colosseum = "/home/wen.423/Desktop/5g/5g-ai/colosseum-logs/1024-NR-5UE-10011-NORMAL/osu-seclab-oai-secsm-ran-ue-img-1021-srn59-RES139156/pcaps"
        self.dataset_5g_select = "/home/wen.423/Desktop/5g/dataset/5g-select"
        self.dataset_5g_colosseum_2 = "/home/wen.423/Desktop/5g/dataset/5g-colosseum-2"

        self.mobiflow_phoenix = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/phoenix"
        self.mobiflow_5g_spector = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/5g-spector"
        self.mobiflow_mobileinsight = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/mobile-insight"
        self.mobiflow_5g_colosseum = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/5g-colosseum"
        self.mobiflow_5g_select = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/5g-select"
        self.mobiflow_5g_colosseum_2 = "/home/wen.423/Desktop/5g/5g-ai/mobiflow/5g-colosseum-2"

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
        if "5g-colosseum" in self.dataset_name or "5g-select" in self.dataset_name:
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

        elif dataset_name == "5g-select":
            mf_folder = self.mobiflow_5g_select
            mf_folder = os.path.join(mf_folder, "benign")
            type = "benign"
            forbidden_set = []
            label = 0

        elif dataset_name == "5g-colosseum-2":
            mf_folder = self.mobiflow_5g_colosseum_2
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
        
        fs = os.listdir(mf_folder)
        out_f = open(f"./data/{dataset_name}_{type}_mobiflow.csv", "w")
        out_f.write(self.mobiflow_meta_str + "\n")
        for f in fs:
            if f in forbidden_set:
                continue
            f = os.path.join(mf_folder, f)
            print(f)

            with open(f, "r") as d:
                for line in d.readlines():
                    if line.startswith("BS"):
                        continue
                    out_f.write(line)

        out_f.close()
            


    def get_mobiflow_index(self, meta_name):
        return self.mobiflow_meta_data.index(meta_name)
        


ver = 5
dl = DataLoader()
# dl.load_data("mobileinsight", False, ver)
# dl.load_data("mobileinsight", True, ver)
# dl.load_data("phoenix", True, ver)
# dl.load_data("5g-spector", True, ver)
# dl.load_data("5g-spector", False, ver)
# dl.load_data("mobileinsight-all", False, ver)
# dl.load_data("5g-colosseum", True, ver)
# dl.load_data("5g-colosseum", False, ver)
# dl.load_data("5g-select", False, ver)
# dl.load_data("5g-colosseum-2", True, ver)
# dl.load_data("5g-colosseum-2", False, ver)

dl.load_data("5g-benign", False, ver)
dl.load_data("5g-attack", True, ver)

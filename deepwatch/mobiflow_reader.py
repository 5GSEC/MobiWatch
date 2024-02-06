import logging
import threading
import datetime
import time
from mobiflow.lockutil import *
from mobiflow.mobiflow import get_time_ms, MOBIFLOW_DELIMITER
from rpc.client import MobiFlowRpcClient

class MobiFlowReader:
    def __init__(self, rpc_ip_addr, rpc_port, query_interval, maintenance_time_threshold=0):
        # input writer
        self.ue_mobiflow_table_name = "ue_mobiflow"
        self.bs_mobiflow_table_name = "bs_mobiflow"
        # init RPC client
        logging.info(f"[App] MobiFlow RPC server config {rpc_ip_addr}:{rpc_port} interval {query_interval}")
        self.rpc_client = MobiFlowRpcClient(rpc_ip_addr, rpc_port)
        self.rpc_query_interval = query_interval  # in ms
        # start a new thread to read database from MobiFlow Auditor xApp
        logging.info("[App] Starting MobiFlow reading thread")
        self.db_thread = threading.Thread(target=self.read_mobiflow_rpc)
        self.db_thread.start()
        # mobiflow storage
        self.bs_mf = {}
        self.ue_mf = {}
        # use dict to track mobiflow entries that have been analyzed
        self.ue_mf_current_index = {}
        self.bs_mf_current_index = {}

    def read_mobiflow_rpc(self):
        if self.rpc_client is None:
            logging.error(f"[App] RPC Client is NULL!")
            return
        if not self.rpc_client.check_server():
            return
        # polling loop
        while True:
            # query MobiFlow from RPC server
            bs_results = self.rpc_client.query_mobiflow_streaming(self.bs_mobiflow_table_name)
            ue_results = self.rpc_client.query_mobiflow_streaming(self.ue_mobiflow_table_name)

            # write MobiFlow based on timestamp order
            u_idx = 0
            b_idx = 0
            while u_idx < len(ue_results) or b_idx < len(bs_results):
                if u_idx >= len(ue_results):
                    write_decision = "BS"
                elif b_idx >= len(bs_results):
                    write_decision = "UE"
                else:
                    # compare timestamp
                    umf = str(ue_results[u_idx])
                    bmf = str(bs_results[b_idx])
                    umf_ts = float(umf.split(MOBIFLOW_DELIMITER)[2])
                    bmf_ts = float(bmf.split(MOBIFLOW_DELIMITER)[2])
                    if umf_ts < bmf_ts:
                        write_decision = "UE"
                    else:
                        write_decision = "BS"

                if write_decision == "UE":
                    umf = ue_results[u_idx]
                    logging.info("[MobiFlow] Storing UE MobiFlow: " + umf)
                    # Store MobiFlow
                    self.add_ue_mobiflow(umf)
                    u_idx += 1
                elif write_decision == "BS":
                    bmf = bs_results[b_idx]
                    logging.info("[MobiFlow] Storing BS MobiFlow: " + bmf)
                    # Store MobiFlow
                    self.add_bs_mobiflow(bmf)
                    b_idx += 1

            time.sleep(self.rpc_query_interval / 1000)
        f.close()

    # add BS mobiflow to corresponding dict
    def add_bs_mobiflow(self, bmf: str):
        bs_id = int(bmf.split(MOBIFLOW_DELIMITER)[5])
        if bs_id in self.bs_mf.keys():
            self.bs_mf[bs_id].append(bmf)
        else:
            self.bs_mf[bs_id] = [bmf]
            self.bs_mf_current_index[bs_id] = 0

    # add UE mobiflow to corresponding dict
    def add_ue_mobiflow(self, umf: str):
        rnti = int(umf.split(MOBIFLOW_DELIMITER)[6])
        if rnti in self.ue_mf.keys():
            self.ue_mf[rnti].append(umf)
        else:
            self.ue_mf[rnti] = [umf]
            self.ue_mf_current_index[rnti] = 0

    # loop through each UE to return the mobiflow that are not analyzed
    def get_next_ue_mobiflow(self, threshold=0):
        rnti_keys = self.ue_mf.keys()
        for rnti in rnti_keys:
            ue_mf_len = len(self.ue_mf[rnti])
            if ue_mf_len <= threshold:
                continue # skip if under threshold
            cur_index = self.ue_mf_current_index[rnti]
            if cur_index < ue_mf_len:
                res = self.ue_mf[rnti][cur_index: ue_mf_len]
                self.ue_mf_current_index[rnti] = ue_mf_len
                return res, rnti
        return None, None

    # loop through each BS to return the mobiflow that are not analyzed
    def get_next_bs_mobiflow(self, threshold=0):
        bs_id_keys = self.bs_mf.keys()
        for bs_id in bs_id_keys:
            bs_mf_len = len(self.bs_mf[bs_id])
            if bs_mf_len <= threshold:
                continue # skip if under threshold
            cur_index = self.bs_mf_current_index[bs_id]
            if cur_index < bs_mf_len:
                res = self.bs_mf[bs_id][cur_index: bs_mf_len]
                self.bs_mf_current_index[bs_id] = bs_mf_len
                return res, bs_id
        return None, None

    @staticmethod
    def timestamp2str(ts):
        return datetime.datetime.fromtimestamp(ts/1000).__str__() # convert ms into s


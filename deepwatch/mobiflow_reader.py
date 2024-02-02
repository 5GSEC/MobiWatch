import logging
import threading
import datetime
import time
from .mobiflow.lockutil import *
from .mobiflow.mobiflow import get_time_ms, MOBIFLOW_DELIMITER
from .rpc.client import MobiFlowRpcClient

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
                    logging.info("[MobiFlow] Writing UE MobiFlow: " + umf)
                    # TODO DO STH... Store MobiFlow
                    u_idx += 1
                elif write_decision == "BS":
                    bmf = bs_results[b_idx]
                    logging.info("[MobiFlow] Writing BS MobiFlow: " + bmf)
                    # TODO DO STH... Store MobiFlow
                    b_idx += 1

            time.sleep(self.rpc_query_interval / 1000)
        f.close()

    @staticmethod
    def timestamp2str(ts):
        return datetime.datetime.fromtimestamp(ts/1000).__str__() # convert ms into s


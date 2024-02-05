#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# SPDX-FileCopyrightText: 2019-present Open Networking Foundation <info@opennetworking.org>
#
# SPDX-License-Identifier: Apache-2.0

# import onos_ric_sdk_py as sdk
import argparse
import json
import logging
from time import sleep
from mobiflow_reader import MobiFlowReader
from dlagent import DeepLogAgent

mf_reader = None
pb = None

def main(args: argparse.Namespace) -> None:
    # load configs
    rpc_ip = args.rpc_addr
    rpc_port = args.rpc_port
    if args.mobiflow_config is not None:
        with open(args.mobiflow_config) as f:
            mobiflow_config = json.load(f)
            db_path = mobiflow_config["mobiflow"]["sqlite3_db_path"]
            rpc_query_interval = mobiflow_config["mobiflow"]["rpc_query_interval"]
    else:
        rpc_query_interval = 500    

    if args.config_path is not None:
        with open(args.config_path) as f:
            general_config = json.load(f)
            num_candidates = int(general_config["deeplog_param"]["num_candidates"])
            window_size = int(general_config["deeplog_param"]["window_size"])
    else:
        num_candidates = 2
        window_size = 5

    # Init mobiflow writer configs
    global mf_reader
    mf_reader = MobiFlowReader(rpc_ip, rpc_port, rpc_query_interval)

    # begin test code
    # with open("example.mobiflow", "r") as f:
    #     for line in f.readlines():
    #         mf_reader.ue_mf.append(line.strip())
    # end test code

    # Loop prediction for incoming data
    deeplog = DeepLogAgent(num_candidates=num_candidates, window_size=window_size)
    last_ue_mf_index = 0
    prediction_interval = rpc_query_interval

    while True:
        # obtain a trace list of MobiFlow and predict
        ue_mf_len = len(mf_reader.ue_mf)
        if last_ue_mf_index < ue_mf_len:
            input_seq = mf_reader.ue_mf[last_ue_mf_index: ue_mf_len]
            last_ue_mf_index = len(mf_reader.ue_mf)-1
            seqs, labels = deeplog.seq_gen(input_seq)
            for i in range(len(seqs)):
                predicted = deeplog.predict(seqs[i])
                deeplog.interpret(seqs[i], predicted, labels[i])

        sleep(prediction_interval / 1000)        

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MobiExpert xApp.")
    parser.add_argument("--config-path", type=str, help="General config path")
    parser.add_argument("--mobiflow-config", type=str, help="MobiFlow config path")
    parser.add_argument("--rpc-addr", type=str, help="RPC server IP address for reaching MobiFlow Service")
    parser.add_argument("--rpc-port", type=int, help="RPC server port for reaching MobiFlow Service")
    args = parser.parse_args()
    main(args)





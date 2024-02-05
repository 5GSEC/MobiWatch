#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# SPDX-FileCopyrightText: 2019-present Open Networking Foundation <info@opennetworking.org>
#
# SPDX-License-Identifier: Apache-2.0

# import onos_ric_sdk_py as sdk
import argparse
import json
import logging
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
        num_candidates = 1
        window_size = 5

    # Init mobiflow writer configs
    global mf_reader
    mf_reader = MobiFlowReader(rpc_ip, rpc_port, rpc_query_interval)

    # begin test code
    # mf_reader.ue_mf.append("UE;0;1700502310.172734000;v2.0;SECSM;0;12927;0;0;0;0;0;3;RRCSetupRequest;0;0;0;0;0;0;0;0")
    # mf_reader.ue_mf.append("UE;1;1700502310.174590000;v2.0;SECSM;0;12927;0;0;0;0;0;3;RRCSetup;2;0;0;0;1700502310.174590000;0;0;0")
    # mf_reader.ue_mf.append("UE;2;1700502310.267372000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;RRCSetupComplete;2;0;0;0;1700502310.174590000;0;0;0")
    # mf_reader.ue_mf.append("UE;3;1700502310.267372000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Registrationrequest;2;1;0;0;1700502310.174590000;0;1700502310.267372000;0")
    # mf_reader.ue_mf.append("UE;4;1700502310.285077000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Authenticationrequest;2;1;0;0;1700502310.174590000;0;1700502310.267372000;0")
    # mf_reader.ue_mf.append("UE;5;1700502310.330620000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Authenticationrequest;2;1;0;0;1700502310.174590000;0;1700502310.267372000;0")
    # mf_reader.ue_mf.append("UE;6;1700502310.361322000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Authenticationresponse;2;1;0;0;1700502310.174590000;0;1700502310.267372000;0")
    # mf_reader.ue_mf.append("UE;7;1700502310.362573000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Securitymodecommand;2;1;0;0;1700502310.174590000;0;1700502310.267372000;0")
    # mf_reader.ue_mf.append("UE;8;1700502310.409600000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Securitymodecommand;2;1;0;0;1700502310.174590000;0;1700502310.267372000;0")
    # mf_reader.ue_mf.append("UE;9;1700502310.431859000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Securitymodecomplete;2;1;1;0;1700502310.174590000;0;1700502310.267372000;0")
    # mf_reader.ue_mf.append("UE;10;1700502310.431859000;v2.0;SECSM;0;12927;0;2089900007487;0;0;0;3;Identityrequest;2;1;1;0;1700502310.174590000;0;1700502310.267372000;0")
    # end test code

    # Loop prediction for incoming data
    deeplog = DeepLogAgent(num_candidates=num_candidates, window_size=window_size)
    last_ue_mf_index = -1
    prediction_interval = rpc_query_interval

    while last_ue_mf_index < len(mf_reader.ue_mf):
        # obtain a trace list of MobiFlow and predict
        input_seq = mf_reader.ue_mf[last_ue_mf_index+1: len(mf_reader.ue_mf)]
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




# ==================================================================================
#
#       Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==================================================================================
import requests
import os
import json
from os import getenv
from ricxappframe.xapp_frame import RMRXapp, rmr
from .utils.constants import Constants
from .manager import *
from .handler import *
from .ai.dlagent import DeepLogAgent, AutoEncoderAgent, LSTMAgent, AutoEncoderAgent_v2, LSTMAgent_v2
from mdclogpy import Level

class MobiWatchXapp:

    __XAPP_CONFIG_PATH = getenv("CONFIG_FILE", default="/tmp/init/config-file.json")
    __XAPP_NAME = "mobiwatch-xapp"
    __XAPP_VERSION = "0.0.1"
    __XAPP_NAME_SPACE = "ricxapp"
    __PLT_NAME_SPACE = "ricplt"
    __HTTP_PORT = 8080
    __RMR_PORT = 4560
    __XAPP_HTTP_END_POINT = "service-%s-%s-http.%s:%d" % (__XAPP_NAME_SPACE, __XAPP_NAME, __XAPP_NAME_SPACE, __HTTP_PORT)
    __XAPP_RMR_END_POINT = "service-%s-%s-rmr.%s:%d" % (__XAPP_NAME_SPACE, __XAPP_NAME, __XAPP_NAME_SPACE, __RMR_PORT)
    __CONFIG_PATH = "/ric/v1/config"

    def __init__(self):
        fake_sdl = getenv("USE_FAKE_SDL", False)
        self._rmr_xapp = RMRXapp(self._default_handler,
                                 config_handler=self._handle_config_change,
                                 rmr_port=self.__RMR_PORT,
                                 post_init=self._post_init,
                                 use_fake_sdl=bool(fake_sdl))

    def _post_init(self, rmr_xapp):
        """
        Function that runs when xapp initialization is complete
        """
        rmr_xapp.logger.set_level(Level.INFO)
        rmr_xapp.logger.info("HWXapp.post_init :: post_init called")
        sdl_mgr = SdlManager(rmr_xapp)
        sub_mgr = SubscriptionManager(rmr_xapp)
        # sdl_alarm_mgr = SdlAlarmManager(rmr_xapp)
        # a1_mgr = A1PolicyManager(rmr_xapp)
        # a1_mgr.startup()
        # metric_mgr = MetricManager(rmr_xapp)
        # metric_mgr.send_metric()

        # register the xApp to the RIC manager
        self._register(rmr_xapp)

        # load xApp config
        models = []
        with open(self.__XAPP_CONFIG_PATH, 'r') as config_file:
            config_json = json.loads(config_file.read())
            models = config_json["mobiwatch"]["models"]
            rmr_xapp.logger.info(f"Loaded xApp model config: {models}")

        for model in models.keys():
            model_path = models[model]["path"]
            model_name = models[model]["name"]
            if model_name == "autoencoder_v2":
                self.dl_ae2_agent = AutoEncoderAgent_v2(model_path)
                # load mobiflow data
                ue_mf, bs_mf = self.dl_ae2_agent.load_mobiflow(sdl_mgr)
                if len(ue_mf) <= 0:
                    return
                seq, df = self.dl_ae2_agent.encode(ue_mf)
                if seq is not None and len(seq) > 0:
                    labels = self.dl_ae2_agent.predict(seq)
                    self.dl_ae2_agent.interpret(df, labels)

            elif model_name == "lstm_v2":
                sqeuence_length = 6 # use the first 5 to predict the 6th
                self.dl_lstm2_agent = LSTMAgent_v2(model_path, sqeuence_length)
                # load mobiflow data
                ue_mf, bs_mf = self.dl_lstm2_agent.load_mobiflow(sdl_mgr)
                if len(ue_mf) <= 0:
                    return
                x_seq, y_seq, df = self.dl_lstm2_agent.encode(ue_mf)
                if x_seq is not None and len(x_seq) > 0:
                    labels = self.dl_lstm2_agent.predict(x_seq, y_seq)
                    self.dl_lstm2_agent.interpret(df, labels)
            
            else:
                rmr_xapp.logger.error(f"Unknown model name: {model_name}")
                continue


        # init DL agent
        # model = "AE"
        # if model == "DeelLog":
        #     model_path = os.path.join(f"/tmp/LSTM_onehot_{train_dataset}_{train_label}_{train_ver}.pth.tar")
        #     self.dl_agent = DeepLogAgent(model_path=model_path, window_size=5, ranking_metric="probability", prob_threshold=0.40)
        #     # load mobiflow data
        #     ue_mf, bs_mf = self.dl_agent.load_mobiflow(sdl_mgr)
        #     if len(ue_mf) <= 0:
        #         return
        #     x, y = self.dl_agent.encode_mobiflow(ue_mf)
        #     for i in range(len(x)):
        #         predict_y = self.dl_agent.predict(x[i])
        #         self.dl_agent.interpret(x[i], predict_y, y[i])
        # if model == "AE":
        #     model_path = os.path.join(f"/tmp/autoencoder_model.pth")
        #     seq_len = 6
        #     self.dl_agent = AutoEncoderAgent(model_path, seq_len)
        #     # load mobiflow data
        #     ue_mf, bs_mf = self.dl_agent.load_mobiflow(sdl_mgr)
        #     if len(ue_mf) <= 0:
        #         return
        #     seq, df = self.dl_agent.encode(ue_mf)
        #     if seq is not None and len(seq) > 0:
        #         labels = self.dl_agent.predict(seq)
        #         self.dl_agent.interpret(df, labels)
        # elif model == "lstm":
        #     model_path = os.path.join(f"/tmp/lstm_multivariate_5g-mobiwatch_benign.pth.tar")
        #     seq_len = 6
        #     self.dl_agent = LSTMAgent(model_path, seq_len)
        #     # load mobiflow data
        #     ue_mf, bs_mf = self.dl_agent.load_mobiflow(sdl_mgr)
        #     if len(ue_mf) <= 0:
        #         return
        #     seq, df = self.dl_agent.encode(ue_mf)
        #     if seq is not None and len(seq) > 0:
        #         labels = self.dl_agent.predict(seq)
        #         self.dl_agent.interpret(df, labels)

    def _register(self, rmr_xapp):
        """
        Register the xApp to the App manager of the near-RT RIC

        Parameters:
            rmr_xapp: instance of RMRXapp
        """
        url = "http://service-%s-appmgr-http.%s:8080/ric/v1/register" % (self.__PLT_NAME_SPACE, self.__PLT_NAME_SPACE)
        with open(self.__XAPP_CONFIG_PATH, 'r') as config_file:
            config_json_str = config_file.read()
        body = {
            "appName": self.__XAPP_NAME,
            "httpEndpoint": self.__XAPP_HTTP_END_POINT,
            "rmrEndpoint": self.__XAPP_RMR_END_POINT,
            "appInstanceName": self.__XAPP_NAME,
            "appVersion": self.__XAPP_VERSION,
            "configPath": self.__CONFIG_PATH,
            "config": config_json_str
        }
        try:
            rmr_xapp.logger.info(f"Sending registration request to {url} {body}")
            response = requests.post(url, json=body)
            rmr_xapp.logger.info(f"Registration response {response.status_code} {response.text}")
            if response.status_code == 201:  # registration request success
                rmr_xapp.logger.info(f"Registration success")

        except IOError as err_h:
            rmr_xapp.logger.error("An IO Error occurred:" + repr(err_h))

    def _deregister(self, rmr_xapp):
        """
        Deregister the xApp to the App manager of the near-RT RIC

        Parameters:
            rmr_xapp: instance of RMRXapp
        """
        url = "http://service-%s-appmgr-http.%s:8080/ric/v1/deregister" % (self.__PLT_NAME_SPACE, self.__PLT_NAME_SPACE)
        body = {
            "appName": self.__XAPP_NAME,
            "appInstanceName": f"{self.__XAPP_NAME}_{self.__XAPP_VERSION}",
        }
        try:
            rmr_xapp.logger.info(f"Sending deregistration request to {url}")
            response = requests.post(url, json=body)
            rmr_xapp.logger.info(f"Deregistration response {response.status_code} {response.text}")
            if response.status_code == 201:  # registration request success
                rmr_xapp.logger.info(f"Deregistration success")

        except IOError as err_h:
            rmr_xapp.logger.error("An IO Error occurred:" + repr(err_h))

    def _handle_config_change(self, rmr_xapp, config):
        """
        Function that runs at start and on every configuration file change.
        """
        rmr_xapp.logger.info("HWXapp.handle_config_change:: config: {}".format(config))
        rmr_xapp.config = config  # No mutex required due to GIL

    def _default_handler(self, rmr_xapp, summary, sbuf):
        """
        Function that processes messages for which no handler is defined
        """
        rmr_xapp.logger.info("HWXapp.default_handler called for msg type = " +
                                   str(summary[rmr.RMR_MS_MSG_TYPE]))
        rmr_xapp.rmr_free(sbuf)

    def createHandlers(self):
        """
        Function that creates all the handlers for RMR Messages
        """
        HealthCheckHandler(self._rmr_xapp, Constants.RIC_HEALTH_CHECK_REQ)
        A1PolicyHandler(self._rmr_xapp, Constants.A1_POLICY_REQ)
        SubscriptionHandler(self._rmr_xapp, Constants.SUBSCRIPTION_REQ)

    def start(self, thread=False):
        """
        This is a convenience function that allows this xapp to run in Docker
        for "real" (no thread, real SDL), but also easily modified for unit testing
        (e.g., use_fake_sdl). The defaults for this function are for the Dockerized xapp.
        """
        self.createHandlers()
        self._rmr_xapp.run(thread)

    def stop(self):
        """
        can only be called if thread=True when started
        TODO: could we register a signal handler for Docker SIGTERM that calls this?
        """
        self._rmr_xapp.stop()




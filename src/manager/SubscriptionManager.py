# ==================================================================================
#
#       Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved.
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
import json
import threading
import http.server
import socketserver
from ricxappframe.xapp_frame import RMRXapp
from ._BaseManager import _BaseManager
from mdclogpy import Level


class SubscriptionManager(_BaseManager):

    __namespace = "e2Manager"
    __SUBSCRIPTION_URL = "http://service-ricplt-submgr-http.ricplt:8088/ric/v1/subscriptions/"
    __CLIENT_END_POINT = "service-ricxapp-template-xapp-http.ricxapp"

    def __init__(self, rmr_xapp: RMRXapp, local_address="0.0.0.0", http_port=8080, rmr_port=4560, report_period=1000) -> None:
        """
        Constructor function
        """
        super().__init__(rmr_xapp)
        self.gnb_list = []
        self.enb_list = []
        self.logger.set_level(Level.INFO)
        self.subscription_list = {}
        self.local_address = local_address
        self.client_http_port = http_port
        self.client_rmr_port = rmr_port
        self.report_period = report_period
        # subscription response
        self.subscription_resp_handler = None
        self.responseCB = None
        self.regsiter_subscription_handler(response_cb=self.handle_subscription_response)

    def query_subscriptions(self):
        """
        Function to obtain the subscription list from the E2 manager
        """
        try:
            response = requests.get(self.__SUBSCRIPTION_URL)

            if response.status_code == 201:  # subscription request success
                resp = json.loads(response.text)
                self.logger.info(f"Subscription query succeeded, response: {resp}")
            else:
                self.logger.error(f"Subscription query failed {response.status_code} {response.text} ")

        except requests.exceptions.HTTPError as err_h:
            return "An Http Error occurred:" + repr(err_h)
        except requests.exceptions.ConnectionError as err_c:
            return "An Error Connecting to the API occurred:" + repr(err_c)
        except requests.exceptions.Timeout as err_t:
            return "A Timeout Error occurred:" + repr(err_t)
        except requests.exceptions.RequestException as err:
            return "An Unknown Error occurred" + repr(err)

    def send_subscription_request(self):
        """
        Function to send subscription request to an E2 node
        """
        subscription_params = self.get_subscription_params()

        self.logger.info(f"Sending subscription request: {self.__SUBSCRIPTION_URL}")
        try:
            response = requests.post(self.__SUBSCRIPTION_URL, json=subscription_params)

            self.logger.info(f"Subscription response {response.status_code} {response.text}")
            if response.status_code == 201:  # subscription request success
                resp = json.loads(response.text)

        except requests.exceptions.HTTPError as err_h:
            return "An Http Error occurred:" + repr(err_h)
        except requests.exceptions.ConnectionError as err_c:
            return "An Error Connecting to the API occurred:" + repr(err_c)
        except requests.exceptions.Timeout as err_t:
            return "A Timeout Error occurred:" + repr(err_t)
        except requests.exceptions.RequestException as err:
            return "An Unknown Error occurred" + repr(err)

    def get_subscription_params(self):
        """
        Function to construct subscription parameters
        """
        # TODO: construct subscription parameters, example format below (fill your own values):
        """
        {"ClientEndPoint": {"Host": self.__CLIENT_END_POINT,
                            "HTTPPort": self.client_http_port,
                            "RMRPort": self.client_rmr_port
                            },
         "Meid": me_id,
         "RANFunctionID": rf_id,
         "E2SubscriptionDirectives": {
             "E2TimeoutTimerValue": 10,
             "E2RetryCount": 2,
             "RMRRoutingNeeded": True
         },
         "SubscriptionDetails": [{"XappEventInstanceID": 1,
                                  "EventTriggers": "event trigger payload",
                                  "ActionToBeSetupList": [
                                      {"ActionID": 0,
                                       "ActionType": "report",
                                       "ActionDefinition": "binary payload of action definition",
                                       "SubsequentActionType": "continue",
                                       "TimeToWait": "w10ms"}
                                  ]}]
         }
        """
        raise NotImplementedError

    def regsiter_subscription_handler(self, response_cb=None):
        """
        Function to register callback handler after subscription request
        This should be called at the constructor
        """
        # Create the thread HTTP server
        if self.subscription_resp_handler is None:
            # Create the server handler if not provided
            class SubscriptionRespHandler(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    # Check if the request path is /ric/v1/subscriptions/response
                    if self.path == "/ric/v1/subscriptions/response":
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        if response_cb:
                            response_cb(post_data)
                        self.send_response(200)
                        self.end_headers()
                    else:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b"404 Not Found")

            class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
                pass

            self.subscription_resp_handler = ThreadedHTTPServer((self.local_address, self.client_http_port), SubscriptionRespHandler)

            # Start the server in a separate thread
            server_thread = threading.Thread(target=self.subscription_resp_handler.serve_forever)
            server_thread.daemon = True  # Terminate thread when main program exits
            server_thread.start()

        if self.subscription_resp_handler is not None:
            if response_cb is not None:
                self.responseCB = response_cb
            return True
        else:
            return False

    def handle_subscription_response(self, resp_data):
        if resp_data is None:
            self.logger.info(f"Empty subscription response received")
            return

        resp_json = json.loads(resp_data)
        self.logger.info(f"Handling subscription response: {resp_json}")


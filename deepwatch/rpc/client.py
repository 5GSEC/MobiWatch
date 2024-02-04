import grpc
import logging
from .protos import mobiflow_service_pb2_grpc
from .protos import mobiflow_service_pb2

class MobiFlowRpcClient:
    def __init__(self, rpc_ip, rpc_port):
        self.rpc_ip = rpc_ip
        self.rpc_port = rpc_port
        self.request_initiator = "DeepWatch"

    def check_server(self):
        if self.rpc_ip is None or self.rpc_ip == "":
            logging.error(f"Invalid RPC address f{self.rpc_ip}!")
            return False
        if self.rpc_port is None or self.rpc_port < 0:
            logging.error(f"Invalid RPC address f{self.rpc_port}!")
            return False
        return True

    def query_mobiflow_streaming(self, table_name) -> list:
        try:
            with grpc.insecure_channel(f'{self.rpc_ip}:{self.rpc_port}') as channel:
                stub = mobiflow_service_pb2_grpc.MobiFlowQueryStub(channel)
                resps = stub.MobiFlowStream(mobiflow_service_pb2.MobiFlowStreamRequest(name=self.request_initiator,
                                                                                        table=table_name))
                mfs = []
                for resp in resps:
                    mfs.append(resp.message)
                return mfs
        except grpc.RpcError as e:
            logging.info(f"[MobiFlowRpcClient] RPC error: {str(e)}")
            return []



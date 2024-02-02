from common import *

# version 1: just use the packet count as feature
class Feature:
    def __init__(self) -> None:
        # packet counts
        # RRC
        self.feature = {
            # "time": 0, # in seconds
            "pkt_cnt_RRCConnectionRequest": 0,
            "pkt_cnt_RRCConnectionSetup": 0,
            "pkt_cnt_RRCConnectionSetupComplete": 0,
            "pkt_cnt_RRCConnectionReject": 0,
            "pkt_cnt_RRCConnectionReestablishment": 0,
            "pkt_cnt_RRCConnectionReestablishmentReject": 0,
            "pkt_cnt_RRCConnectionReestablishmentRequest": 0,
            "pkt_cnt_RRCConnectionReestablishmentComplete": 0,
            "pkt_cnt_RRCConnectionRelease": 0,
            "pkt_cnt_SecurityModeComplete": 0,
            "pkt_cnt_SecurityModeFailure": 0,
            "pkt_cnt_UECapabilityInformation": 0,
            "pkt_cnt_ULInformationTransfer": 0,
            "pkt_cnt_DLInformationTransfer": 0,
            "pkt_cnt_RRCConnectionReconfiguration": 0,
            "pkt_cnt_RRCConnectionReconfigurationComplete": 0,
            "pkt_cnt_SecurityModeCommand": 0,
            "pkt_cnt_UECapabilityEnquiry": 0,
            "pkt_cnt_RRCConnectionResume-r13": 0,
            "pkt_cnt_MeasurementReport": 0,
            "pkt_cnt_ATTACH_REQUEST": 0,
            "pkt_cnt_ATTACH_ACCEPT": 0,
            "pkt_cnt_ATTACH_COMPLETE": 0,
            "pkt_cnt_ATTACH_REJECT": 0,
            "pkt_cnt_DETACH_REQUEST": 0,
            "pkt_cnt_DETACH_ACCEPT": 0,
            "pkt_cnt_TRACKING_AREA_UPDATE_REQUEST": 0,
            "pkt_cnt_TRACKING_AREA_UPDATE_ACCEPT": 0,
            "pkt_cnt_TRACKING_AREA_UPDATE_COMPLETE": 0,
            "pkt_cnt_TRACKING_AREA_UPDATE_REJECT": 0,
            "pkt_cnt_EXTENDED_SERVICE_REQUEST": 0,
            "pkt_cnt_SERVICE_REJECT": 0,
            "pkt_cnt_GUTI_REALLOCATION_COMMAND": 0,
            "pkt_cnt_GUTI_REALLOCATION_COMPLETE": 0,
            "pkt_cnt_AUTHENTICATION_REQUEST": 0,
            "pkt_cnt_AUTHENTICATION_RESPONSE": 0,
            "pkt_cnt_AUTHENTICATION_REJECT": 0,
            "pkt_cnt_AUTHENTICATION_FAILURE": 0,
            "pkt_cnt_IDENTITY_REQUEST": 0,
            "pkt_cnt_IDENTITY_RESPONSE": 0,
            "pkt_cnt_SECURITY_MODE_COMMAND": 0,
            "pkt_cnt_SECURITY_MODE_COMPLETE": 0,
            "pkt_cnt_SECURITY_MODE_REJECT": 0,
            "pkt_cnt_EMM_STATUS": 0,
            "pkt_cnt_EMM_INFORMATION": 0,
            "pkt_cnt_DOWNLINK_NAS_TRANSPORT": 0,
            "pkt_cnt_UPLINK_NAS_TRANSPORT": 0,
            "pkt_cnt_CS_SERVICE_NOTIFICATION": 0,
            "pkt_cnt_SERVICE_REQUEST": 0,
            # "label": 0
        }

    def add(self, feature_name):
        if feature_name in self.feature.keys():
            self.feature[feature_name] += 1
        else:
            # print("Feature %s not found" % feature_name)
            return
    
    def clear_counter(self):
        for k in self.feature:
            self.feature[k] = 0

    # def set_label(self, val):
    #     self.feature["label"] = val

    def _get_feature_desc(self):
        desc = {}
        for i in range(self.feature.keys().__len__()):
            desc[i] = list(self.feature.keys())[i]
        return desc

    def to_np_array(self):
        return np.array(list(self.feature.values()))

    def encode(self, trace_list, label):
        data = np.empty(shape=(0, self.feature.keys().__len__()))
        labels = np.ones(self.feature.keys().__len__()) * label
        feature_prefix = "pkt_cnt_"
        for trace in trace_list:
            for m in trace:
                self.add(feature_prefix + m)
            data = np.append(data, [self.to_np_array()], axis=0)
            self.clear_counter()
        return data, labels, self._get_feature_desc()

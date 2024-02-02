from common import *

# version 2: use a 2D matrix to encode both packet count and message context
class FeatureV2:
    def __init__(self) -> None:
        self.keys = [
            "RRCConnectionRequest",
            "RRCConnectionSetup",
            "RRCConnectionSetupComplete",
            "RRCConnectionReject",
            "RRCConnectionReestablishment",
            #"RRCConnectionReestablishmentReject",
            "RRCConnectionReestablishmentRequest",
            #"RRCConnectionReestablishmentComplete",
            "RRCConnectionRelease",
            "SecurityModeCommand",
            "SecurityModeComplete",
            "SecurityModeFailure",
            "UECapabilityEnquiry",
            "UECapabilityInformation",
            # "ULInformationTransfer",
            # "DLInformationTransfer",
            "RRCConnectionReconfiguration",
            #"RRCConnectionReconfigurationComplete": 0,
            # "RRCConnectionResume-r13",
            "ATTACH_REQUEST",
            # "ATTACH_ACCEPT",
            # "ATTACH_COMPLETE",
            "ATTACH_REJECT",
            "DETACH_REQUEST",
            "DETACH_ACCEPT",
            "TRACKING_AREA_UPDATE_REQUEST",
            # "TRACKING_AREA_UPDATE_ACCEPT",
            # "TRACKING_AREA_UPDATE_COMPLETE",
            "TRACKING_AREA_UPDATE_REJECT",
            # "EXTENDED_SERVICE_REQUEST",
            "SERVICE_REJECT",
            # "GUTI_REALLOCATION_COMMAND",
            # "GUTI_REALLOCATION_COMPLETE",
            "AUTHENTICATION_REQUEST",
            "AUTHENTICATION_RESPONSE",
            "AUTHENTICATION_REJECT",
            "AUTHENTICATION_FAILURE",
            "IDENTITY_REQUEST",
            "IDENTITY_RESPONSE",
            "SECURITY_MODE_COMMAND",
            # "SECURITY_MODE_COMPLETE",
            "SECURITY_MODE_REJECT",
            # "EMM_STATUS",
            # "EMM_INFORMATION",
            # "DOWNLINK_NAS_TRANSPORT",
            # "UPLINK_NAS_TRANSPORT",
            # "CS_SERVICE_NOTIFICATION",
            "SERVICE_REQUEST",
        ]
        self.keylen = self.keys.__len__()

        self.data = np.zeros(shape=(self.keylen, self.keylen))
    
    def _get_msg_index(self, m):
        return self.keys.index(m)
    
    def _get_feature_desc(self):
        desc = {}
        cnt = 0
        for i in range(self.keys.__len__()):
            for j in range(self.keys.__len__()):
                if i == j:
                    desc[cnt] = "cnt_" + self.keys[i]
                else:
                    desc[cnt] = "dist_%s_%s" % (self.keys[i], self.keys[j]) 
                cnt += 1
        return desc

    def encode(self, trace_list, label):
        data = np.empty(shape=(0, self.keylen * self.keylen))
        for trace in trace_list:
            vec = self._encode_single(trace)
            data = np.append(data, [vec], axis=0)
        labels = np.ones(data.shape[0]) * label
        return data, labels, self._get_feature_desc()

    def _encode_single(self, trace: list):
        # prune the trace list first
        trace_pruned = []
        for m in trace:
            if m in self.keys:
                trace_pruned.append(m)
        # print(trace_pruned)

        matrix = np.zeros(shape=(self.keylen, self.keylen))
        count = np.zeros(self.keylen)
        l = trace_pruned.__len__()
        for i in range(l):
            msg_i = trace_pruned[i]
            index_i = self._get_msg_index(msg_i)
            count[index_i] += 1
            for j in range(l):
                msg_j = trace_pruned[j]
                index_j = self._get_msg_index(msg_j)
                if i == j:
                    pass
                else:
                    dist = np.abs(j - i)
                    # print("dist %s %s %d" % (msg_i, msg_j, dist))
                    matrix[index_i][index_j] += dist

        # average
        sum_cnt = np.sum(count)
        for i in range(self.keylen):
            for j in range(self.keylen):
                if i == j:
                    # assign frequency
                    if sum_cnt != 0:
                        matrix[i][j] = count[i] / sum_cnt
                    else:
                        matrix[i][j] = 0
                else:
                    # average
                    if count[i] != 0:
                        matrix[i][j] = matrix[i][j] / count[i]

        # print(matrix.shape)
        # print(matrix.flatten().shape)
        return matrix.flatten()



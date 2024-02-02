from common import *

# Version 3, treat it as a NLP sentence and use N-gram to encode it
class FeatureV3:
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

        self.vocab = []

        # Set start and end gram
        self.start_gram = 2
        self.end_gram = 2
        self.msg_threshold = 2
        self.feature_desc = {}

        self.data = None
    
    def get_msg_index(self, m):
        return self.keys.index(m)
    
    def _get_vocabulary(self):
        vocab = []
        for k1 in self.keys:
            for k2 in self.keys:
                v = "%s %s" % (k1, k2)
                if v not in self.vocab:
                    vocab.append(v)
        return vocab
    
    def get_feature_description(self):
        return self.feature_desc

    def encode(self, trace_list, label):    
        # prune the trace list first
        trace_list_pruned = []
        trace_pruned = []
        for trace in trace_list:
            for m in trace:
                if m in self.keys:
                    trace_pruned.append(m)
            if trace_pruned.__len__() > self.msg_threshold:
                # we filter trace with a threshold to avoid short trace
                trace_list_pruned.append(" ".join(trace_pruned)) # convert it to a string
            trace_pruned = []

        # init vocabulary
        self.vocab = self._get_vocabulary()
        
        # start at bigrams and end at bigrams
        vectorizer_ngram = CountVectorizer(ngram_range=(self.start_gram, self.end_gram), lowercase=False, vocabulary=self.vocab)
        try:
            vector_ngram = vectorizer_ngram.fit_transform(trace_list_pruned)
        except:
            print(trace_list_pruned)
            return
        
        # print(self.keys.__len__())
        # construct feature description
        for (k, v) in vectorizer_ngram.vocabulary_.items():
            self.feature_desc[v] = k

        data = vector_ngram.toarray()
        # print(trace_list_pruned)
        # np.set_printoptions(threshold=np.inf)
        # print(data)
        labels = np.zeros(data.shape[0]) * label

        return data, labels, self.get_feature_description()

        

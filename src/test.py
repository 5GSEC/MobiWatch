from .ai.dlagent import DeepLogAgent
from .ai.deeplog.msgseq import MsgSeq
from .ai.dlagent import AutoEncoderAgent

# init DL agent
mf_trace = [
    "UE;0;1715451191330.7598;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;RRCSetupRequest;0;0;0;0;0;0;0;0",
    "UE;1;1715451191330.815;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;RRCSetup;2;0;0;0;1715451191330.7192;0;0;0",
    "UE;2;1715451191330.9485;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;RRCSetupComplete;2;0;0;0;1715451191330.7192;0;0;0",
    "UE;3;1715451191330.973;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;Registrationrequest;2;1;0;0;1715451191330.7192;0;1715451191330.7192;0",
    "UE;4;1715451191331.0315;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;Authenticationrequest;2;1;0;0;1715451191330.7192;0;1715451191330.7192;0",
    "UE;5;1715451191331.053;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;Authenticationresponse;2;1;0;0;1715451191330.7192;0;1715451191330.7192;0",
    "UE;6;1715451191331.0737;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;Securitymodecommand;2;1;0;0;1715451191330.7192;0;1715451191330.7192;0",
    "UE;7;1715451191331.0967;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;Securitymodecomplete;2;1;0;0;1715451191330.7192;0;1715451191330.7192;0",
    "UE;8;1715451191331.1292;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;SecurityModeCommand;2;1;0;0;1715451191330.7192;0;1715451191330.7192;0",
    "UE;9;1715451191331.1736;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;SecurityModeComplete;2;1;1;0;1715451191330.7192;0;1715451191330.7192;0",
    # "UE;10;1715451191331.2073;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;RRCReconfiguration;2;1;1;0;1715451191330.7192;0;1715451191330.7192;0",
    # "UE;11;1715451191331.229;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;RRCReconfigurationComplete;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192",
    # "UE;12;1715451191331.2688;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;UECapabilityEnquiry;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192",
    # "UE;13;1715451191331.29;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;UECapabilityInformation;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192",
    # "UE;14;1715451192331.1719;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;Registrationcomplete;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192",
    # "UE;15;1715451192331.2004;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;ULNAStransport;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192",
    # "UE;16;1715451193350.4912;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;RRCReconfiguration;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192",
    # "UE;17;1715451193350.6807;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;DLNAStransport;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192",
    # "UE;18;1715451193350.704;v2.0;SECSM;0;8331;1450744508;0;0;2;2;0;RRCReconfigurationComplete;2;2;1;0;1715451191330.7192;0;1715451191330.7192;1715451191330.7192"
]

# DeepLog
# dl_agent = DeepLogAgent(model_path="/home/wen.423/Desktop/5g/5g-ai/5G-DeepWatch/deepwatch/ai/deeplog/save/LSTM_onehot_5g-select_benign_v5.pth.tar", window_size=5, ranking_metric="probability", prob_threshold=0.40)
# msg_seq = MsgSeq()
# x, y = msg_seq.encode(mf_trace, window_size=5)

# for i in range(len(x)):
#     predict_y = dl_agent.predict(x[i])
#     dl_agent.interpret(x[i], predict_y, y[i])


# AE
dl_agent = AutoEncoderAgent(model_path="/home/wen.423/Desktop/5g/osc/MobiWatch/src/ai/autoencoder/data/autoencoder_model.pth", sequence_length=6)
mf_dict = {}
for i in range(len(mf_trace)):
    mf_dict[i] = mf_trace[i]
seq, df = dl_agent.encode(mf_dict)
print(seq.shape)
labels = dl_agent.predict(seq)
print(labels)
dl_agent.interpret(df, labels)

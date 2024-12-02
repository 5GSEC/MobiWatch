from .ai.deeplog.msgseq import MsgSeq
from .ai.dlagent import DeepLogAgent, AutoEncoderAgent, LSTMAgent

# init DL agent
mf_trace = [
    "UE;0;v2.0;SECSM;1733109001;12345678;1;52979;52979;1450744508;0;0;0;2;RRCSetupRequest; ;0;0;0;3;0;0",
    "UE;1;v2.0;SECSM;1733109001;12345678;1;52979;52979;1450744508;0;0;0;2;RRCSetup; ;2;0;0;0;0;0",
    "UE;2;v2.0;SECSM;1733109001;12345678;1;52979;52979;1450744508;0;0;0;2;RRCSetupComplete;Registrationrequest;2;0;0;1;0;0",
    "UE;3;v2.0;SECSM;1733109002;12345678;1;52979;52979;1450744508;2;2;0;2;DLInformationTransfer;Authenticationrequest;2;0;0;0;0;0",
    "UE;4;v2.0;SECSM;1733109002;12345678;1;52979;52979;1450744508;2;2;0;2;ULInformationTransfer;Authenticationresponse;2;0;0;0;0;0",
    "UE;5;v2.0;SECSM;1733109002;12345678;1;52979;52979;1450744508;2;2;0;2;DLInformationTransfer;Securitymodecommand;2;0;0;0;0;0",
    "UE;6;v2.0;SECSM;1733109002;12345678;1;52979;52979;1450744508;2;2;0;2;ULInformationTransfer;Securitymodecomplete;2;0;0;0;0;0",
    "UE;7;v2.0;SECSM;1733109002;12345678;1;52979;52979;1450744508;2;2;0;2;SecurityModeCommand; ;2;0;0;0;0;0",
    "UE;8;v2.0;SECSM;1733109002;12345678;1;52979;52979;1450744508;2;2;0;2;SecurityModeComplete; ;2;0;3;0;0;0",
    "UE;9;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;RRCSetupRequest; ;0;0;0;3;0;0",
    "UE;10;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;RRCSetup; ;2;0;0;0;0;0",
    "UE;11;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;RRCSetupComplete;Registrationrequest;2;0;0;2;0;0",
    "UE;12;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;DLInformationTransfer;Identityrequest;2;0;0;0;0;0",
    "UE;13;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;ULInformationTransfer;Identityresponse;2;0;0;0;0;0",
    "UE;14;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;DLInformationTransfer;Authenticationrequest;2;0;0;0;0;0",
    "UE;15;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;ULInformationTransfer;Authenticationresponse;2;0;0;0;0;0",
    "UE;16;v2.0;SECSM;1733109018;12345678;2;58701;58701;1450744508;0;0;0;2;DLInformationTransfer;Registrationreject;2;0;0;0;0;0"
]

# DeepLog
# dl_agent = DeepLogAgent(model_path="/home/wen.423/Desktop/5g/5g-ai/5G-DeepWatch/deepwatch/ai/deeplog/save/LSTM_onehot_5g-select_benign_v5.pth.tar", window_size=5, ranking_metric="probability", prob_threshold=0.40)
# msg_seq = MsgSeq()
# x, y = msg_seq.encode(mf_trace, window_size=5)

# for i in range(len(x)):
#     predict_y = dl_agent.predict(x[i])
#     dl_agent.interpret(x[i], predict_y, y[i])


# AE
dl_agent = AutoEncoderAgent(model_path="./src/ai/autoencoder/data/autoencoder_model.pth", sequence_length=6)
mf_dict = {}
for i in range(len(mf_trace)):
    mf_dict[i] = mf_trace[i]
seq, df = dl_agent.encode(mf_dict)
print(seq.shape)
labels = dl_agent.predict(seq)
print(labels)
dl_agent.interpret(df, labels)

# LSTM
# dl_agent = LSTMAgent(model_path="./src/ai/lstm/save/lstm_multivariate_5g-mobiwatch_benign.pth.tar", sequence_length=6)
# mf_dict = {}
# for i in range(len(mf_trace)):
#     mf_dict[i] = mf_trace[i]
# seq, df = dl_agent.encode(mf_dict)
# print(seq.shape)
# labels = dl_agent.predict(seq)
# print(labels)
# dl_agent.interpret(df, labels)

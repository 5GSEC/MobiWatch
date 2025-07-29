from .ai.deeplog.msgseq import MsgSeq
from .ai.dlagent import LSTMAgent_v2, AutoEncoderAgent_v2

# init DL agent
mf_trace_benign = [
    "UE;0;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;RRCSetupRequest; ;0;0;0;3;0;0",
    "UE;1;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;RRCSetup; ;2;0;0;0;0;0",
    "UE;2;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;RRCSetupComplete;Registrationrequest;2;1;0;1;0;0",
    "UE;3;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;DLInformationTransfer;Authenticationrequest;2;1;0;0;0;0",
    "UE;4;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;ULInformationTransfer;Authenticationresponse;2;1;0;0;0;0",
    "UE;5;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;DLInformationTransfer;Securitymodecommand;2;1;0;0;0;0",
    "UE;6;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;ULInformationTransfer;Securitymodecomplete;2;1;0;0;0;0",
    "UE;7;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;SecurityModeCommand; ;2;1;0;0;0;0",
    "UE;8;v2.1;SECSM;1749482829;10000;1;2880;2880;0;2089900004778;2;2;0;2;SecurityModeComplete; ;2;1;3;0;0;0",
]

mf_trace_abnormal = [
    "UE;27;v2.1;SECSM;1749482846;10000;4;53243;53243;0;0;0;0;0;0;RRCSetupRequest; ;0;0;0;3;0;0",
    "UE;28;v2.1;SECSM;1749482846;10000;4;53243;53243;0;0;0;0;0;0;RRCSetup; ;2;0;0;0;0;0",
    "UE;29;v2.1;SECSM;1749482847;10000;4;53243;53243;0;2089900004709;2;2;0;2;RRCSetupComplete;Registrationrequest;2;1;0;1;0;0",
    "UE;30;v2.1;SECSM;1749482847;10000;4;53243;53243;0;2089900004709;2;2;0;2;DLInformationTransfer;Authenticationrequest;2;1;0;0;0;0",
    "UE;31;v2.1;SECSM;1749482847;10000;4;53243;53243;0;2089900004709;2;2;0;2;ULInformationTransfer;Authenticationresponse;2;1;0;0;0;0",
    "UE;32;v2.1;SECSM;1749482847;10000;4;53243;53243;0;2089900004709;2;2;0;2;DLInformationTransfer;Securitymodecommand;2;1;0;0;0;0",
    "UE;33;v2.1;SECSM;1749482847;10000;4;53243;53243;0;2089900004709;2;2;0;2;ULInformationTransfer;Securitymodecomplete;2;1;0;0;0;0",
    "UE;34;v2.1;SECSM;1749482847;10000;4;53243;53243;0;2089900004709;2;2;0;2;SecurityModeCommand; ;2;1;0;0;0;0",
    "UE;35;v2.1;SECSM;1749482847;10000;4;53243;53243;0;2089900004709;2;2;0;2;SecurityModeComplete; ;2;1;3;0;0;0",
    "UE;36;v2.1;SECSM;1749482848;10000;4;53243;53243;0;2089900004709;2;2;0;2;SecurityModeComplete; ;2;1;3;0;0;0",
    "UE;37;v2.1;SECSM;1749482848;10000;4;53243;53243;0;2089900004709;2;2;0;2;RRCRelease; ;1;0;0;0;0;0",
    "UE;38;v2.1;SECSM;1749482848;10000;4;53243;53243;0;2089900004709;2;2;0;2;RRCReconfigurationComplete; ;1;0;0;0;0;0",
]

mf_trace = mf_trace_benign


# AE
# dl_agent = AutoEncoderAgent_v2(model_path="./src/ai/autoencoder_v2/data/autoencoder_v2_model.pth")
# mf_dict = {}
# for i in range(len(mf_trace)):
#     mf_dict[i] = mf_trace[i]
# seq, df = dl_agent.encode(mf_dict)
# print(seq.shape)
# labels = dl_agent.predict(seq)
# print(labels)
# dl_agent.interpret(df, labels)

# LSTM
dl_agent = LSTMAgent_v2(model_path="./src/ai/lstm_v2/data/lstm_multivariate_mobiflow_v2_benign.pth.tar", sequence_length=6)
mf_dict = {}
for i in range(len(mf_trace)):
    mf_dict[i] = mf_trace[i]
x_seq, y_seq, df = dl_agent.encode(mf_dict)
print(x_seq.shape)
labels = dl_agent.predict(x_seq, y_seq)
print(labels)
dl_agent.interpret(df, labels)

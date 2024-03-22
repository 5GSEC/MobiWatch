
# NAS_LTE
nas_emm_code = {
65: "ATTACH_REQUEST",
66: "ATTACH_ACCEPT",
67: "ATTACH_COMPLETE",
68: "ATTACH_REJECT",
69: "DETACH_REQUEST",
70: "DETACH_ACCEPT",
72: "TRACKING_AREA_UPDATE_REQUEST",
73: "TRACKING_AREA_UPDATE_ACCEPT",
74: "TRACKING_AREA_UPDATE_COMPLETE",
75: "TRACKING_AREA_UPDATE_REJECT",
76: "EXTENDED_SERVICE_REQUEST",
78: "SERVICE_REJECT",
80: "GUTI_REALLOCATION_COMMAND",
81: "GUTI_REALLOCATION_COMPLETE",
82: "AUTHENTICATION_REQUEST",
83: "AUTHENTICATION_RESPONSE",
84: "AUTHENTICATION_REJECT",
92: "AUTHENTICATION_FAILURE",
85: "IDENTITY_REQUEST",
86: "IDENTITY_RESPONSE",
93: "SECURITY_MODE_COMMAND",
94: "SECURITY_MODE_COMPLETE",
95: "SECURITY_MODE_REJECT",
96: "EMM_STATUS",
97: "EMM_INFORMATION",
98: "DOWNLINK_NAS_TRANSPORT",
99: "UPLINK_NAS_TRANSPORT",
100: "CS_SERVICE_NOTIFICATION",
77: "SERVICE_REQUEST"}


# RRC_LTE
rrc_ul_dcch_code = {
1: "CSFBParametersRequestCDMA2000",
2: "MeasurementReport",
3: "RRCConnectionReconfigurationComplete",
4: "RRCConnectionReestablishmentComplete",
5: "RRCConnectionSetupComplete",
6: "SecurityModeComplete",
7: "SecurityModeFailure",
8: "UECapabilityInformation",
9: "ULHandoverPreparationTransfer",
10: "ULInformationTransfer",
11: "CounterCheckResponse",
12: "UEInformationResponse-r9",
13: "ProximityIndication-r9",
14: "RNReconfigurationComplete-r10",
15: "MBMSCountingResponse-r10",
16: "InterFreqRSTDMeasurementIndication-r10"}

rrc_ul_ccch_code = {1: "RRCConnectionReestablishmentRequest",
2: "RRCConnectionRequest"}

rrc_dl_dcch_code = {1: "CSFBParametersResponseCDMA2000",
2: "DLInformationTransfer",
3: "HandoverFromEUTRAPreparationRequest",
4: "MobilityFromEUTRACommand",
5: "RRCConnectionReconfiguration",
6: "RRCConnectionRelease",
7: "SecurityModeCommand",
8: "UECapabilityEnquiry",
9: "CounterCheck",
10: "UEInformationRequest-r9",
11: "LoggedMeasurementConfiguration-r10",
12: "RNReconfiguration-r10",
13: "RRCConnectionResume-r13",}


rrc_dl_ccch_code = {1: "RRCConnectionReestablishment",
2: "RRCConnectionReestablishmentReject",
3: "RRCConnectionReject",
4: "RRCConnectionSetup"}


# NAS_NR: openair3/NAS/COMMON/NR_NAS_defs.h
nas_emm_code_NR = {
0x41: 'Registrationrequest',
0x42: 'Registrationaccept',
0x43: 'Registrationcomplete',
0x44: 'Registrationreject',
0x45: 'DeregistrationrequestUEoriginating',
0x46: 'DeregistrationacceptUEoriginating',
0x47: 'DeregistrationrequestUEterminated',
0x48: 'DeregistrationacceptUEterminated',
0x4c: 'Servicerequest',
0x4d: 'Servicereject',
0x4e: 'Serviceaccept',
0x4f: 'Controlplaneservicerequest',
0x50: 'Networkslicespecificauthenticationcommand',
0x51: 'Networkslicespecificauthenticationcomplete',
0x52: 'Networkslicespecificauthenticationresult',
0x54: 'Configurationupdatecommand',
0x55: 'Configurationupdatecomplete',
0x56: 'Authenticationrequest',
0x57: 'Authenticationresponse',
0x58: 'Authenticationreject',
0x59: 'Authenticationfailure',
0x5a: 'Authenticationresult',
0x5b: 'Identityrequest',
0x5c: 'Identityresponse',
0x5d: 'Securitymodecommand',
0x5e: 'Securitymodecomplete',
0x5f: 'Securitymodereject',
0x64: 'SGMMstatus',
0x65: 'Notification',
0x66: 'Notificationresponse',
0x67: 'ULNAStransport',
0x68: 'DLNAStransport',
0xc1: 'PDUsessionestablishmentrequest',
0xc2: 'PDUsessionestablishmentaccept',
0xc3: 'PDUsessionestablishmentreject',
0xc5: 'PDUsessionauthenticationcommand',
0xc6: 'PDUsessionauthenticationcomplete',
0xc7: 'PDUsessionauthenticationresult',
0xc9: 'PDUsessionmodificationrequest',
0xca: 'PDUsessionmodificationreject',
0xcb: 'PDUsessionmodificationcommand',
0xcc: 'PDUsessionmodificationcomplete',
0xcd: 'PDUsessionmodificationcommandreject',
0xd1: 'PDUsessionreleaserequest',
0xd2: 'PDUsessionreleasereject',
0xd3: 'PDUsessionreleasecommand',
0xd4: 'PDUsessionreleasecomplete',
0xd6: 'SGSMstatus'
}


# RRC_NR: openair2/RRC/NR/MESSAGES/ASN.1/nr-rrc-17.3.0.asn1
rrc_dl_ccch_code_NR = {
1: "RRCReject",
2: "RRCSetup"
}

rrc_dl_dcch_code_NR = {
1: "RRCReconfiguration",
2: "RRCResume",
3: "RRCRelease",
4: "RRCReestablishment",
5: "SecurityModeCommand",
6: "DLInformationTransfer",
7: "UECapabilityEnquiry",
8: "CounterCheck",
9: "MobilityFromNRCommand",
10: "DLDedicatedMessageSegment-r16",
11: "UEInformationRequest-r16",
12: "DLInformationTransferMRDC-r16",
13: "LoggedMeasurementConfiguration-r16"
}

rrc_ul_ccch_code_NR = {
1: "RRCSetupRequest",
2: "RRCResumeRequest",
3: "RRCReestablishmentRequest",
4: "RRCSystemInfoRequest",
}

rrc_ul_dcch_code_NR = {
1: "MeasurementReport",
2: "RRCReconfigurationComplete",
3: "RRCSetupComplete",
4: "RRCReestablishmentComplete",
5: "RRCResumeComplete",
6: "SecurityModeComplete",
7: "SecurityModeFailure",
8: "ULInformationTransfer",
9: "LocationMeasurementIndication",
10: "UECapabilityInformation",
11: "CounterCheckResponse",
12: "UEAssistanceInformation",
13: "FailureInformation",
14: "ULInformationTransferMRDC",
15: "SCGFailureInformation",
16: "SCGFailureInformationEUTRA",
}


def decode_rrc_msg(dcch: int, downlink: int, msg_id: int, rat: int) -> str:
    if dcch == 1 and downlink == 1:
        if rat == 0:
            return rrc_dl_dcch_code.get(msg_id)
        elif rat == 1:
            return rrc_dl_dcch_code_NR.get(msg_id)
        else:
            return ""
    elif dcch == 1 and downlink == 0:
        if rat == 0:
            return rrc_ul_dcch_code.get(msg_id)
        elif rat == 1:
            return rrc_ul_dcch_code_NR.get(msg_id)
        else:
            return ""
    elif dcch == 0 and downlink == 1:
        if rat == 0:
            return rrc_dl_ccch_code.get(msg_id)
        elif rat == 1:
            return rrc_dl_ccch_code_NR.get(msg_id)
        else:
            return ""
    else:
        if rat == 0:
            return rrc_ul_ccch_code.get(msg_id)
        elif rat == 1:
            return rrc_ul_ccch_code_NR.get(msg_id)
        else:
            return ""


def decode_nas_msg(dis: int, msg_id: int, rat: int) -> str:
    if rat == 0:  # LTE
        if dis == 1:
            return nas_emm_code.get(msg_id)
        else:
            # TODO implement ESM
            return ""
    elif rat == 1:  # NR
        if dis == 0:  # EMM = 0x7e & 1 = 0
            return nas_emm_code_NR.get(msg_id)
        else:
            # TODO implement ESM
            return ""

    return ""


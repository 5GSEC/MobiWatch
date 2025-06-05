import time
import logging
import copy
from enum import Enum
from typing import List
from .encoding import *

def get_time_ms():
    return time.time() * 1000

def get_time_sec():
    return time.time()

###################### Auxiliary classes ######################

class State(Enum):
    def __str__(self):
        return str(self.value)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

class RRCState(State):
    INACTIVE = 0
    RRC_IDLE = 1
    RRC_CONNECTED = 2
    RRC_RECONGIFURED = 3


class EMMState(State):
    EMM_DEREGISTERED = 0
    EMM_REGISTER_INIT = 1
    EMM_REGISTERED = 2


class SecState(State):
    SEC_CONTEXT_NOT_EXIST = 0
    SEC_CONTEXT_EXIST = 1

###################### Constants ######################

MOBIFLOW_VERSION = "v2.1"
GENERATOR_NAME = "SECSM"
UE_MOBIFLOW_ID_COUNTER = 0
BS_MOBIFLOW_ID_COUNTER = 0
MOBIFLOW_DELIMITER = ";"
UE_META_DATA_ITEM_LEN = 0
UE_MOBIFLOW_ITEM_LEN = 0

###################### MobiFlow Structures ######################

class UEMobiFlow:
    def __init__(self):
        self.msg_type = "UE"                        # Msg hdr  - mobiflow type [UE, BS]
        self.msg_id = 0                             # Msg hdr  - unique mobiflow event ID
        self.mobiflow_ver = MOBIFLOW_VERSION        # Msg hdr  - version of Mobiflow
        self.generator_name = GENERATOR_NAME        # Msg hdr  - generator name (e.g., SECSM)
        #####################################################################
        self.timestamp = 0              # UE meta  - timestamp (ms)
        self.nr_cell_id = 0             # UE meta  - NR (5G) basestation id
        self.gnb_cu_ue_f1ap_id = 0      # UE meta  - UE id identified by gNB CU F1AP
        self.gnb_du_ue_f1ap_id = 0      # UE meta  - UE id identified by gNB DU F1AP
        self.rnti = 0                   # UE meta  - ue rnti
        self.s_tmsi = 0                 # UE meta  - ue s-tmsi
        self.mobile_id = 0              # UE meta  - mobile device id (e.g., SUPI, SUCI, IMEI)
        self.rrc_cipher_alg = 0         # UE packet telemetry  - rrc cipher algorithm
        self.rrc_integrity_alg = 0      # UE packet telemetry  - rrc integrity algorithm
        self.nas_cipher_alg = 0         # UE packet telemetry  - nas cipher algorithm
        self.nas_integrity_alg = 0      # UE packet telemetry  - nas integrity algorithm
        #####################################################################
        self.rrc_msg = ""               # UE packet-agnostic telemetry  - RRC message
        self.nas_msg = ""               # UE packet-agnostic telemetry  - NAS message
        self.rrc_state = 0              # UE packet-agnostic telemetry  - RRC state       [INACTIVE, RRC_IDLE, RRC_CONNECTED, RRC_RECONFIGURED]
        self.nas_state = 0              # UE packet-agnostic telemetry  - NAS state (EMM) [EMM_DEREGISTERED, EMM_REGISTER_INIT, EMM_REGISTERED]
        self.rrc_sec_state = 0          # UE packet-agnostic telemetry  - security state  [SEC_CONTEXT_NOT_EXIST, SEC_CONTEXT_EXIST]
        #####################################################################
        self.reserved_field_1 = 0       # UE packet-specific telemetry
        self.reserved_field_2 = 0       # UE packet-specific telemetry
        self.reserved_field_3 = 0       # UE packet-specific telemetry
        #####################################################################
        # self.rrc_initial_timer = 0      # UE timer  -
        # self.rrc_inactive_timer = 0     # UE timer  -
        # self.nas_initial_timer = 0      # UE timer  -
        # self.nas_inactive_timer = 0     # UE timer  -

    def __str__(self):
        attrs = []
        for a in self.__dict__.values():
            if str(a) == "":
                attrs.append(" ")  # avoid parse error in C
            else:
                attrs.append(str(a))
        return MOBIFLOW_DELIMITER.join(attrs)
    
    def copy(self):
        # Return a deep copy of the current object
        return copy.deepcopy(self)

class BSMobiFlow:
    def __init__(self):
        self.msg_type = "BS"            # Msg hdr  - mobiflow type [UE, BS]
        self.msg_id = 0                 # Msg hdr  - unique mobiflow event ID
        self.timestamp = get_time_sec()              # Msg hdr  - timestamp (ms)
        self.mobiflow_ver = MOBIFLOW_VERSION        # Msg hdr  - version of Mobiflow
        self.generator_name = GENERATOR_NAME        # Msg hdr  - generator name (e.g., SECSM)
        #####################################################################
        self.nr_cell_id = 0             # BS meta  - basestation id
        self.mcc = ""                   # BS meta  - mobile country code
        self.mnc = ""                   # BS meta  - mobile network code
        self.tac = ""                   # BS meta  - tracking area code
        self.report_period = 0          # BS meta  - report period (ms)
        #####################################################################
        self.connected_ue_cnt = 0       # BS stats -
        self.idle_ue_cnt = 0            # BS stats -
        self.max_ue_cnt = 0             # BS stats -
        #####################################################################
        self.initial_timer = 0          # BS timer  -
        self.inactive_timer = 0         # BS timer  -

    def __str__(self):
        attrs = []
        for a in self.__dict__.values():
            if str(a) == "":
                attrs.append(" ")  # avoid parse error in C
            else:
                attrs.append(str(a))
        return MOBIFLOW_DELIMITER.join(attrs)


###################### SECSM Structures ######################
class UE:
    def __init__(self) -> None:
        #### UE Meta ####
        self.timestamp = 0
        self.nr_cell_id = 0
        self.gnb_cu_ue_f1ap_id = 0
        self.gnb_du_ue_f1ap_id = 0
        self.rnti = 0
        self.s_tmsi = 0
        #### UE Attributes ####
        self.rrc_cipher_alg = 0
        self.rrc_integrity_alg = 0
        self.nas_cipher_alg = 0
        self.nas_integrity_alg = 0
        self.rrc_state = RRCState.INACTIVE
        self.nas_state = EMMState.EMM_DEREGISTERED
        self.rrc_sec_state = SecState.SEC_CONTEXT_NOT_EXIST
        self.emm_cause = 0
        #### UE Timer ####
        self.rrc_initial_timer = 0
        self.rrc_inactive_timer = 0
        self.nas_initial_timer = 0
        self.nas_inactive_timer = 0
        #### History record ####
        self.msg_trace = []
        self.current_msg_index = 0    # indicate how many messages are reported to PBEST
        # UE mobiflow report criteria: report new msg
        self.should_report = True     # indicate whether the UE state has changed and should be reported
        self.last_ts = 0              # indicate timestamp last time the UE is updated

    def __str__(self) -> str:
        s = "UE: {gnb_du_ue_f1ap_id: %s, rnti: %s, s_tmsi: %s}\n" % (self.gnb_du_ue_f1ap_id, hex(self.rnti), hex(self.s_tmsi))
        m = "MsgTrace: \n"
        for msg in self.msg_trace:
            m = m + msg + "\n"
        return s + m

    def __eq__(self, other) -> bool:
        if not isinstance(other, UE):
            return False
        return self.nr_cell_id == other.nr_cell_id and self.rnti == other.rnti

    def propagate(self, msg: str):
        # implement state transition and counter update
        prev_rrc_state = self.rrc_state
        prev_nas_state = self.nas_state
        prev_rrc_sec_state = self.rrc_sec_state
        if msg == "RRCConnectionResume-r13" or msg == "RRCConnectionSetup" or msg == "RRCConnectionReestablishment"\
                or msg == "RRCResume" or msg == "RRCSetup" or msg == "RRCReestablishment":
            if self.rrc_state == RRCState.INACTIVE or self.rrc_state == RRCState.RRC_IDLE:
                self.rrc_state = RRCState.RRC_CONNECTED

        elif msg == "RRCConnectionRelease" \
                or msg == "RRCRelease":
            self.rrc_state = RRCState.RRC_IDLE
            self.nas_state = EMMState.EMM_DEREGISTERED
            self.rrc_sec_state = SecState.SEC_CONTEXT_NOT_EXIST

        elif msg == "RRCConnectionReject" or msg == "RRCConnectionReestablishmentReject"\
                or msg == "RRCReject":
            self.rrc_state = RRCState.RRC_IDLE

        elif msg == "ATTACH_REQUEST" or msg == "RRCConnectionReestablishmentComplete"\
                or msg == "Registrationrequest" or msg == "Servicerequest":
            self.nas_state = EMMState.EMM_REGISTER_INIT

        elif msg == "SERVICE_REQUEST": # LTE: The UE shall treat that the service request procedure as successful, when the lower layers indicate that the user plane radio bearer is successfully set up
            self.nas_state = EMMState.EMM_REGISTERED

        elif msg == "SecurityModeComplete": # RRC security state
                # or msg == "Securitymodecomplete":
            self.rrc_sec_state = SecState.SEC_CONTEXT_EXIST

        elif msg == "ATTACH_COMPLETE"\
                or msg == "RRCReconfigurationComplete" or msg == "Registrationcomplete" or msg == "Serviceaccept":
            self.nas_state = EMMState.EMM_REGISTERED

        elif msg == "ATTACH_REJECT" or msg == "SERVICE_REJECT" or msg == "DETACH_ACCEPT" or msg == "TRACKING_AREA_UPDATE_REJECT" \
                or msg == "Registrationreject" or msg == "Servicereject" or msg == "DeregistrationacceptUEoriginating":
            self.nas_state = EMMState.EMM_DEREGISTERED
            self.rrc_sec_state = SecState.SEC_CONTEXT_NOT_EXIST

        # update timer
        if prev_rrc_state != self.rrc_state:
            if prev_rrc_state < RRCState.RRC_CONNECTED <= self.rrc_state:
                self.rrc_initial_timer = self.last_ts
            if self.rrc_state < RRCState.RRC_CONNECTED <= prev_rrc_state:
                self.rrc_inactive_timer = self.last_ts
        if prev_nas_state != self.nas_state:
            if prev_nas_state <= EMMState.EMM_DEREGISTERED < self.nas_state:
                self.nas_initial_timer = self.last_ts
            if self.nas_state <= EMMState.EMM_DEREGISTERED < prev_nas_state or prev_nas_state < EMMState.EMM_REGISTERED <= self.nas_state:
                self.nas_inactive_timer = self.last_ts
        if prev_rrc_sec_state != self.rrc_sec_state:
            pass

        return prev_rrc_state, prev_nas_state, prev_rrc_sec_state, self.rrc_state, self.nas_state, self.rrc_sec_state

    def generate_mobiflow(self):
        umf = UEMobiFlow()
        # propagate constant
        global UE_MOBIFLOW_ID_COUNTER
        umf.msg_id = UE_MOBIFLOW_ID_COUNTER
        UE_MOBIFLOW_ID_COUNTER += 1
        umf.timestamp = self.timestamp
        umf.nr_cell_id = self.nr_cell_id
        umf.rnti = self.rnti
        umf.s_tmsi = self.s_tmsi
        umf.cipher_alg = self.cipher_alg
        umf.integrity_alg = self.integrity_alg
        umf.emm_cause = self.emm_cause
        prev_rrc, prev_nas, prev_rrc_sec, rrc, nas, sec = 0, 0, 0, 0, 0, 0
        if self.current_msg_index < self.msg_trace.__len__():
            umf.msg = self.msg_trace[self.current_msg_index]
            self.current_msg_index += 1
            prev_rrc, prev_nas, prev_rrc_sec, rrc, nas, sec = self.propagate(umf.msg)  # update UE state
            umf.rrc_state = self.rrc_state
            umf.nas_state = self.nas_state
            umf.rrc_sec_state = self.rrc_sec_state
            umf.rrc_initial_timer = self.rrc_initial_timer
            umf.rrc_inactive_timer = self.rrc_inactive_timer
            umf.nas_initial_timer = self.nas_initial_timer
            umf.nas_inactive_timer = self.nas_inactive_timer
        if self.current_msg_index >= self.msg_trace.__len__():
            self.should_report = False
        return umf, prev_rrc, prev_nas, prev_rrc_sec, rrc, nas, sec

class BS:
    def __init__(self) -> None:
        ####  BS Meta ####
        self.nr_cell_id = 0
        self.mcc = 0
        self.mnc = 0
        self.tac = 0
        self.report_period = 0
        #### BS Stats ####
        self.connected_ue_cnt = 0
        self.idle_ue_cnt = 0
        self.max_ue_cnt = 0
        #### BS Timer ####
        self.initial_timer = get_time_sec()
        self.inactive_timer = 0
        #### History record ####
        self.ue = []
        # BS mobiflow report criteria: change of BS stats / timer
        self.should_report = True
        self.name = ""

    def add_ue(self, ur: UE) -> None:
        # update UE if found
        for u in self.ue:
            if u.__eq__(ur):
                u.last_ts = get_time_sec()
                return

        # add new UE
        ur.last_ts = get_time_sec()
        self.ue.append(ur)

    def __str__(self) -> str:
        uestr = ""
        for u in self.ue:
            uestr = uestr + hex(u.rnti) + " "
        s = "BsRecord: {%d}" % self.nr_cell_id
        return s

    def __eq__(self, other) -> bool:
        if not isinstance(other, BS):
            return False
        return self.nr_cell_id == other.nr_cell_id

    def update_counters(self, prev_rrc, prev_nas, prev_sec, rrc, nas, sec):
        if prev_rrc != rrc:
            if rrc == RRCState.RRC_CONNECTED:
                self.connected_ue_cnt += 1
                self.should_report = True
            if rrc == RRCState.RRC_IDLE:
                self.idle_ue_cnt += 1
                self.should_report = True
            if prev_rrc == RRCState.RRC_CONNECTED and rrc < RRCState.RRC_CONNECTED:
                self.connected_ue_cnt -= 1
            if prev_rrc == RRCState.RRC_IDLE and rrc > RRCState.RRC_IDLE:
                self.idle_ue_cnt -= 1
        if prev_nas != nas:
            self.should_report = True
        if prev_sec != sec:
            self.should_report = True
        if prev_rrc == rrc and prev_nas == nas and prev_sec == sec:
            self.should_report = False

    def generate_mobiflow(self) -> BSMobiFlow:
        bmf = BSMobiFlow()
        # propagate constant
        global BS_MOBIFLOW_ID_COUNTER
        bmf.msg_id = BS_MOBIFLOW_ID_COUNTER
        BS_MOBIFLOW_ID_COUNTER += 1
        bmf.timestamp = get_time_sec()
        bmf.nr_cell_id = self.nr_cell_id
        bmf.mcc = self.mcc
        bmf.mnc = self.mnc
        bmf.tac = self.tac
        bmf.report_period = self.report_period
        bmf.connected_ue_cnt = self.connected_ue_cnt
        bmf.idle_ue_cnt = self.idle_ue_cnt
        bmf.max_ue_cnt = self.max_ue_cnt
        bmf.initial_timer = self.initial_timer
        bmf.inactive_timer = self.inactive_timer
        self.should_report = False
        return bmf


def parse_measurement_into_mobiflow(kpm_measurement_dict: dict) -> List[UEMobiFlow]:
    mfs = []
    kpm_keys = kpm_measurement_dict.keys()

    if "gnb_du_ue_f1ap_id" in kpm_keys and int(kpm_measurement_dict["gnb_du_ue_f1ap_id"]) == 0:
        return mfs # ignore empty indication records
    
    global UE_META_DATA_ITEM_LEN, UE_MOBIFLOW_ITEM_LEN
    if UE_META_DATA_ITEM_LEN == 0 and UE_MOBIFLOW_ITEM_LEN == 0:
        # initialize Meta values
        UE_MOBIFLOW_ITEM_LEN = len(kpm_measurement_dict.keys())
        for k in kpm_keys:
            if not k.startswith("msg"):
                UE_META_DATA_ITEM_LEN = UE_META_DATA_ITEM_LEN + 1

    msg_len = UE_MOBIFLOW_ITEM_LEN - UE_META_DATA_ITEM_LEN

    # populate MobiFlow telemetry
    mf = UEMobiFlow()

    # Metadata telemetry
    for key in kpm_keys:
        if key in mf.__dict__.keys():
            setattr(mf, key, int(kpm_measurement_dict[key])) # assume key name and variable name matches
        elif key == "s_tmsi_part1":
            mf.s_tmsi = int(kpm_measurement_dict[key]) << 32
        elif key == "s_tmsi_part2":
            mf.s_tmsi = mf.s_tmsi + int(kpm_measurement_dict[key])
        elif key == "nr_cell_id_part1":
            mf.nr_cell_id = int(kpm_measurement_dict[key]) << 32
        elif key == "nr_cell_id_part2":
            mf.nr_cell_id = mf.nr_cell_id + int(kpm_measurement_dict[key])
        elif key == "mobile_id_part1":
            mf.mobile_id = int(kpm_measurement_dict[key]) << 32
        elif key == "mobile_id_part2":
            mf.mobile_id = mf.mobile_id + int(kpm_measurement_dict[key])
        elif key.startswith("msg"):
            continue
        else:
            logging.error(f"[MobiFlow] Unknown telemetry {key}")

    # Packet-level telemetry
    for i in range(1, msg_len+1):
        mf = mf.copy() # one MobiFlow entry per packet
        msg_val = int(kpm_measurement_dict[f"msg{i}"])
        if msg_val == 0:
            continue
        rrc_msg_id = (msg_val >> 27) & 0x1F
        dcch_ccch = (msg_val >> 26) & 0x01               # 1 bit
        downlink_uplink = (msg_val >> 25) & 0x01         # 1 bit
        nas_msg_id = (msg_val >> 19) & 0x3F              # 6 bits
        emm_esm = (msg_val >> 18) & 0x01                 # 1 bit
        mf.rrc_state = (msg_val >> 16) & 0x03               # 2 bits
        mf.nas_state = (msg_val >> 14) & 0x03               # 2 bits
        mf.rrc_sec_state = (msg_val >> 12) & 0x03           # 2 bits
        mf.reserved_field_1 = (msg_val >> 8) & 0x0F         # 4 bits
        mf.reserved_field_2 = (msg_val >> 4) & 0x0F         # 4 bits
        mf.reserved_field_3 = (msg_val) & 0x0F              # 4 bits

        rrc_msg_name = decode_rrc_msg(dcch_ccch, downlink_uplink, rrc_msg_id, 1)
        if rrc_msg_name != "" and rrc_msg_name is not None:
            mf.rrc_msg = rrc_msg_name
        elif rrc_msg_id != 0:
            mf.rrc_msg = ""
            logging.error(f"[MobiFlow] Invalid RRC Msg dcch={dcch_ccch}, downlink={downlink_uplink} {rrc_msg_id}")

        if emm_esm != 0:
            # EMM message
            nas_msg_id = nas_msg_id + list(nas_emm_code_NR.keys())[0] # add nas message offset
            nas_msg_name = decode_nas_msg(emm_esm, nas_msg_id, 1)
            if nas_msg_name != "" and nas_msg_name is not None:
                mf.nas_msg = nas_msg_name
            elif nas_msg_id != 0:
                logging.error(f"[MobiFlow] Invalid NAS Msg: discriminator={emm_esm} {nas_msg_id}")
                mf.nas_msg = ""
        else:
            # not EMM message
            mf.nas_msg = ""
        
        global UE_MOBIFLOW_ID_COUNTER
        mf.msg_id = UE_MOBIFLOW_ID_COUNTER
        UE_MOBIFLOW_ID_COUNTER = UE_MOBIFLOW_ID_COUNTER + 1
        mfs.append(mf)

    return mfs
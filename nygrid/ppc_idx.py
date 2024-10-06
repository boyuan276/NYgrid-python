# Import pypower data indexing
from pypower.idx_bus import (BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
                             VM, VA, BASE_KV, ZONE, VMAX, VMIN)
from pypower.idx_gen import (GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE,
                             GEN_STATUS, PMAX, PMIN, RAMP_AGC, RAMP_10, RAMP_30)
from pypower.idx_brch import (F_BUS, T_BUS, BR_R, BR_X, BR_B,
                              RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS)
from pypower.idx_cost import MODEL, STARTUP, SHUTDOWN, NCOST, POLYNOMIAL, COST

# Define index for generator commitment keys
CMT_KEY = 24
MIN_UP_TIME = 25
MIN_DOWN_TIME = 26
OFFLINE = -1
AVAILABLE = 1
MUSTRUN = 2

# Define index for DC lines
DC_NAME = 0
DC_F_BUS = 1
DC_T_BUS = 2
DC_BR_STATUS = 3
DC_PF = 4
DC_PT = 5
DC_QF = 6
DC_QT = 7
DC_VF = 8
DC_VT = 9
DC_PMIN = 10
DC_PMAX = 11
DC_QMINF = 12
DC_QMAXF = 13
DC_QMINT = 14
DC_QMAXT = 15
LOSS0 = 16
LOSS1 = 17

# Define index for ESRs
ESR_NAME = 0
ESR_BUS = 1
ESR_STATUS = 2
ESR_CRG_MAX = 3
ESR_DIS_MAX = 4
ESR_CRG_EFF = 5
ESR_DIS_EFF = 6
ESR_SOC_MIN = 7
ESR_SOC_MAX = 8
ESR_SOC_INI = 9
ESR_SOC_TGT = 10
ESR_CRG_COST = 11
ESR_DIS_COST = 12

# Define index for VREs
VRE_BUS = 0
VRE_PMAX = 1
VRE_PMIN = 2
VRE_TYPE = 3
VRE_NAME = 4

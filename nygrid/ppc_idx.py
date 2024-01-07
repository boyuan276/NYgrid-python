# Import pypower data indexing
from pypower.idx_bus import (BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
                             VM, VA, BASE_KV, ZONE, VMAX, VMIN)
from pypower.idx_gen import (GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE,
                             GEN_STATUS, PMAX, PMIN, RAMP_AGC, RAMP_10, RAMP_30)
from pypower.idx_brch import (F_BUS, T_BUS, BR_R, BR_X, BR_B,
                              RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS)
from pypower.idx_cost import MODEL, NCOST, POLYNOMIAL, COST

# Define index for DC lines
DC_F_BUS = 0
DC_T_BUS = 1
DC_BR_STATUS = 2
DC_PF = 3
DC_PT = 4
DC_QF = 5
DC_QT = 6
DC_VF = 7
DC_VT = 8
DC_PMIN = 9
DC_PMAX = 10
DC_QMINF = 11
DC_QMAXF = 12
DC_QMINT = 13
DC_QMAXT = 14
LOSS0 = 15
LOSS1 = 16

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

from pyomo.environ import (ConcreteModel, Var, Constraint, ConstraintList, Objective,
                           Reals, NonNegativeReals, minimize, Suffix)
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.common.timing import TicTocTimer
import numpy as np
import pandas as pd
import pypower.api as pp
# Import pypower data indexing
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
from . import utlis as utils
import logging
from run_nygrid import NYGrid


class Optimizer:

    def __init__(self, nygrid):
        """
        Initialize the NYGrid model.

        Parameters
        ----------
        nygrid : NYGrid
            NYGrid simulation model.

        Returns
        -------
        None
        """

        # Define pyomo model
        self.nygrid = nygrid
        self.model = ConcreteModel(name='multi-period OPF')

        # User-defined initial condition
        self.gen_init = None
        self.esr_init = None

        # Penalty parameters and default values
        # ED module
        self.NoPowerBalanceViolation = False
        self.NoRampViolation = False
        self.PenaltyForOverGeneration = 1_500  # $/MWh
        self.PenaltyForLoadShed = 5_000  # $/MWh
        self.PenaltyForRampViolation = 11_000  # $/MW
        # UC module
        self.PenaltyForMinTimeViolation = 1_000  # $/MWh
        self.PenaltyForNumberCommitViolation = 10_000  # $/hour
        # RS module
        self.NoReserveViolation = False
        self.PenaltyForReserveViolation = 1_300  # $/MW
        self.NoImplicitReserveCascading = False
        self.OfflineReserveNotFromOnline = False
        # PF module
        self.NoPowerflowViolation = False
        self.SlackBusName = None
        self.SystemBaseMVA = 100  # MVA
        self.HvdcHurdleCost = 0.10  # $/MWh
        self.PenaltyForBranchMwViolation = 1_000  # $/MWh
        self.PenaltyForInterfaceMWViolation = 1_000  # $/MWh
        self.MaxPhaseAngleDifference = 1.5  # Radians
        # ES module
        self.PenaltyForEnergyStorageViolation = 8_000  # $/MWh

    def set_dimensions(self, NT, NB, NG, NL, NDC, NBR, NIF):
        """
        Set the dimensions of the optimization problem.

        Args:
            NT:
            NB:
            NG:
            NL:
            NDC:
            NBR:
            NIF:

        Returns:

        """
        self.NT = NT
        self.NB = NB
        self.NG = NG
        self.NL = NL
        self.NDC = NDC
        self.NBR = NBR
        self.NIF = NIF

    def set_ts_params(self, load_sch, gen_mw_sch, gen_max_sch, gen_min_sch,
                      gen_ramp_sch, gen_cost0_sch, gen_cost1_sch):
        """
        Set the ED module parameters.

        Args:
            load_sch:
            gen_mw_sch:
            gen_max_sch:
            gen_min_sch:
            gen_ramp_sch:
            gen_cost0_sch:
            gen_cost1_sch:

        Returns:

        """
        self.load_sch = load_sch
        self.gen_mw_sch = gen_mw_sch
        self.gen_max_sch = gen_max_sch
        self.gen_min_sch = gen_min_sch
        self.gen_ramp_sch = gen_ramp_sch
        self.gen_cost0_sch = gen_cost0_sch
        self.gen_cost1_sch = gen_cost1_sch

    def set_ed_params(self):
        """
        Set the ED module parameters.

        Args:
            load_pu:
            gen_max:
            gen_min:
            ramp_up:
            ramp_down:
            gencost_0:
            gencost_1:

        Returns:

        """

        self.load_pu = self.nygrid.load_pu
        self.gen_max = self.nygrid.gen_max
        self.gen_min = self.nygrid.gen_min
        self.ramp_up = self.nygrid.ramp_up
        self.ramp_down = self.nygrid.ramp_down
        self.gencost_0 = self.nygrid.gencost_0
        self.gencost_1 = self.nygrid.gencost_1

    def set_uc_params(self, gen_init, esr_init):
        """
        Set the UC module parameters.

        Args:
            gen_init:
            esr_init:

        Returns:

        """
        self.gen_init = gen_init
        self.esr_init = esr_init

    def set_rs_params(self):
        """
        Set the RS module parameters.

        Returns:

        """
        pass

    def set_pf_params(self, Bf, B, gen_map, load_map, if_map, br_max, br_min,
                      if_lims):
        """
        Set the PF module parameters.

        Args:
            Bf:
            B:
            gen_map:
            load_map:
            if_map:
            br_max:
            br_min:
            if_lims:

        Returns:

        """

        self.Bf = Bf
        self.B = B
        self.gen_map = gen_map
        self.load_map = load_map
        self.if_map = if_map
        self.br_max = br_max
        self.br_min = br_min
        self.if_lims = if_lims

    def set_es_params(self):
        """
        Set the ES module parameters.

        Returns:

        """
        pass

    def create_multi_opf_soft_new(self):
        """
        Multi-period OPF problem.

        Parameters:
            A tuple of OPF network parameters and constraints.

        Returns:
            model (pyomo.core.base.PyomoModel.ConcreteModel): Pyomo model for multi-period OPF problem.
        """

        # %% Define variables

        # Generator real power output
        self.model.PG = Var(range(self.NT), range(self.NG),
                            within=Reals, initialize=1)
        # Bus phase angle
        self.model.Va = Var(range(self.NT), range(self.NB),
                            within=Reals, initialize=0,
                            bounds=(-2 * np.pi, 2 * np.pi))
        # Branch power flow
        self.model.PF = Var(range(self.NT), range(self.NBR),
                            within=Reals, initialize=0)

        # Dual variables for price information
        self.model.dual = Suffix(direction=Suffix.IMPORT)

        # Slack variables for soft constraints
        # Slack variable for interface flow upper bound
        self.model.s_if_max = Var(range(self.NT), range(len(self.if_lims)),
                                  within=NonNegativeReals, initialize=0)
        # Slack variable for interface flow lower bound
        self.model.s_if_min = Var(range(self.NT), range(len(self.if_lims)),
                                  within=NonNegativeReals, initialize=0)
        # Slack variable for branch flow upper bound
        self.model.s_br_max = Var(range(self.NT), range(self.NBR),
                                  within=NonNegativeReals, initialize=0)
        # Slack variable for branch flow lower bound
        self.model.s_br_min = Var(range(self.NT), range(self.NBR),
                                  within=NonNegativeReals, initialize=0)

        # %% Define objective function

        # Generator energy cost
        def gen_cost_ene_expr(model, gencost_0, gencost_1):
            return sum(gencost_0[t, g] + gencost_1[t, g] * model.PG[t, g]
                       for g in range(self.NG) for t in range(self.NT))

        # Penalty for violating interface flow limits
        def if_penalty_expr(model, penalty_for_if_violation):
            return sum(model.s_if_max[t, n] + model.s_if_min[t, n]
                       for n in range(len(self.if_lims)) for t in range(self.NT)) * penalty_for_if_violation

        # Penalty for violating branch flow limits
        def br_penalty_expr(model, penalty_for_br_violation):
            return sum(model.s_br_max[t, n] + model.s_br_min[t, n]
                       for n in range(self.NBR) for t in range(self.NT)) * penalty_for_br_violation

        # TODO: Remove slack cost weight and change to penalty for violating interface flow limits
        # TODO: Add penalty for over generation and penalty for load shed
        # TODO: Add penalty for ramp violation
        self.model.obj = Objective(expr=(gen_cost_ene_expr(self.model, self.gencost_0, self.gencost_1)
                                         + if_penalty_expr(self.model, self.PenaltyForInterfaceMWViolation)
                                         + br_penalty_expr(self.model, self.PenaltyForBranchMwViolation)),
                                   sense=minimize)

        # %% Define constraints

        # 1.1. Branch flow definition
        def branch_flow_rule(model, t, br):
            return model.PF[t, br] == sum(self.Bf[br, b] * model.Va[t, b]
                                          for b in range(self.NB))

        self.model.c_br_flow = Constraint(range(self.NT), range(self.NBR),
                                          rule=branch_flow_rule)

        # 1.2. Branch flow upper limit
        def branch_flow_max_rule(model, t, br):
            return model.PF[t, br] <= self.br_max[br] + model.s_br_max[t, br]

        self.model.c_br_max = Constraint(range(self.NT), range(self.NBR),
                                         rule=branch_flow_max_rule)

        # 1.3. Branch flow lower limit
        def branch_flow_min_rule(model, t, br):
            return - model.PF[t, br] <= - self.br_min[br] + model.s_br_min[t, br]

        self.model.c_br_min = Constraint(range(self.NT), range(self.NBR),
                                         rule=branch_flow_min_rule)

        # 2.1. Generator real power output upper limit
        def gen_power_max_rule(model, t, g):
            return model.PG[t, g] <= self.gen_max[t, g]

        self.model.c_gen_max = Constraint(range(self.NT), range(self.NG),
                                          rule=gen_power_max_rule)

        # 2.2. Generator real power output lower limit
        def gen_power_min_rule(model, t, g):
            return - model.PG[t, g] <= - self.gen_min[t, g]

        self.model.c_gen_min = Constraint(range(self.NT), range(self.NG),
                                          rule=gen_power_min_rule)

        # 3.1. Generator ramp rate downward limit
        def gen_ramp_rate_down_rule(model, t, g):
            if t == 0:
                if self.gen_init is not None:
                    return - model.PG[t, g] + self.gen_init[g] <= self.ramp_down[t, g]
                else:
                    return Constraint.Skip
            else:
                return - model.PG[t, g] + model.PG[t - 1, g] <= self.ramp_down[t, g]

        self.model.c_gen_ramp_down = Constraint(range(self.NT), range(self.NG),
                                                rule=gen_ramp_rate_down_rule)

        # 3.2. Generator ramp rate limit
        def gen_ramp_rate_up_rule(model, t, g):
            if t == 0:
                if self.gen_init is not None:
                    return model.PG[t, g] - self.gen_init[g] <= self.ramp_up[t, g]
                else:
                    return Constraint.Skip
            else:
                return model.PG[t, g] - model.PG[t - 1, g] <= self.ramp_up[t, g]

        self.model.c_gen_ramp_up = Constraint(range(self.NT), range(self.NG),
                                              rule=gen_ramp_rate_up_rule)

        # 4.1. DC power flow constraint
        def dc_power_flow_rule(model, t, b):
            return sum(self.gen_map[b, g] * model.PG[t, g] for g in range(self.NG)) \
                - sum(self.load_map[b, l] * self.load_pu[t, l] for l in range(self.NL)) \
                == sum(self.B[b, b_] * model.Va[t, b_] for b_ in range(self.NB))

        self.model.c_pf = Constraint(range(self.NT), range(self.NB),
                                     rule=dc_power_flow_rule)

        # 4.2. DC line power balance constraint
        def dc_line_power_balance_rule(model, t, idx_f, idx_t):
            return model.PG[t, idx_f] == - model.PG[t, idx_t]

        self.model.c_dcline = Constraint(range(self.NT), range(self.NDC),
                                         rule=dc_line_power_balance_rule)

        # 5.1. Interface flow upper limit
        def interface_flow_max_rule(model, t, n):
            if_id, if_lims_min, if_lims_max = self.if_lims[n, :]
            br_dir_idx = self.if_map[(self.if_map[:, 0] == int(if_id)), 1]
            br_dir, br_idx = np.sign(br_dir_idx), np.abs(
                br_dir_idx).astype(int)
            return if_lims_max >= sum(br_dir[i] * model.PF[t, br_idx[i]]
                                      for i in range(len(br_idx))) + model.s_if_max[t, n]

        self.model.c_if_max = Constraint(range(self.NT), range(len(self.if_lims)),
                                         rule=interface_flow_max_rule)

        # 5.2. Interface flow lower limit
        def interface_flow_min_rule(model, t, n):
            if_id, if_lims_min, if_lims_max = self.if_lims[n, :]
            br_dir_idx = self.if_map[(self.if_map[:, 0] == int(if_id)), 1]
            br_dir, br_idx = np.sign(br_dir_idx), np.abs(
                br_dir_idx).astype(int)
            return if_lims_min <= sum(br_dir[i] * model.PF[t, br_idx[i]]
                                      for i in range(len(br_idx))) + model.s_if_min[t, n]

        self.model.c_if_min = Constraint(range(self.NT), range(len(self.if_lims)),
                                         rule=interface_flow_min_rule)

        logging.debug('Created model with soft interface flow limit constraints ...')

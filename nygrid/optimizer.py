import pyomo.environ as pyo
import numpy as np
import logging


class Optimizer:

    def __init__(self, nygrid):
        """
        Initialize the NYGrid model.

        Parameters
        ----------
        nygrid : nygrid.run_nygrid.NYGrid

        Returns
        -------
        None
        """

        # Define pyomo model
        self.nygrid = nygrid

        if nygrid.model is None:
            self.model = pyo.ConcreteModel(name='multi-period DC OPF')
        else:
            self.model = nygrid.model

        # User-defined initial condition
        # TODO: Add initial condition for ES module

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
        self.PenaltyForBranchMwViolation = 1e21  # $/MWh
        self.PenaltyForInterfaceMWViolation = 1e21  # $/MWh
        self.MaxPhaseAngleDifference = 1.5  # Radians
        # ES module
        self.PenaltyForEnergyStorageViolation = 8_000  # $/MWh

        # %% Define sets
        self.times = range(self.nygrid.NT)
        self.buses = range(self.nygrid.NB)
        self.branches = range(self.nygrid.NBR)
        self.generators = range(self.nygrid.NG)
        self.loads = range(self.nygrid.NL)
        self.interfaces = range(self.nygrid.NIF)
        self.dclines = range(self.nygrid.NDCL)

    def add_vars_ed(self):
        """
        Add variables for ED module.

        Returns
        -------
        None
        """

        # Generator real power output
        self.model.PG = pyo.Var(self.times, self.generators,
                                within=pyo.Reals, initialize=1)

        logging.debug('Added variables for ED module.')

        # Slack variable for ramp rate downward limit
        self.model.s_ramp_down = pyo.Var(self.times, self.generators,
                                         within=pyo.NonNegativeReals, initialize=0)

        # Slack variable for ramp rate upward limit
        self.model.s_ramp_up = pyo.Var(self.times, self.generators,
                                       within=pyo.NonNegativeReals, initialize=0)

    def add_vars_pf(self):
        """
        Add variables for PF module.

        Returns
        -------
        None
        """

        # Bus phase angle
        self.model.VA = pyo.Var(self.times, self.buses,
                                within=pyo.Reals, initialize=0,
                                bounds=(-2 * np.pi, 2 * np.pi))
        # Branch power flow
        self.model.PF = pyo.Var(self.times, self.branches,
                                within=pyo.Reals, initialize=0)

        # Slack variable for interface flow upper bound
        self.model.s_if_max = pyo.Var(self.times, self.interfaces,
                                      within=pyo.NonNegativeReals, initialize=0)
        # Slack variable for interface flow lower bound
        self.model.s_if_min = pyo.Var(self.times, self.interfaces,
                                      within=pyo.NonNegativeReals, initialize=0)
        # Slack variable for branch flow upper bound
        self.model.s_br_max = pyo.Var(self.times, self.branches,
                                      within=pyo.NonNegativeReals, initialize=0)
        # Slack variable for branch flow lower bound
        self.model.s_br_min = pyo.Var(self.times, self.branches,
                                      within=pyo.NonNegativeReals, initialize=0)

        logging.debug('Added variables for PF module.')

    def add_vars_uc(self):
        """
        Add variables for UC module.

        Returns
        -------
        None
        """

        raise NotImplementedError('UC module is not implemented yet.')

    def add_vars_rs(self):
        """
        Add variables for RS module.

        Returns
        -------
        None
        """

        raise NotImplementedError('RS module is not implemented yet.')

    def add_vars_es(self):

        raise NotImplementedError('ES module is not implemented yet.')

    def add_vars_dual(self):

        # Dual variables for price information
        self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def add_obj(self):
        """
        Add objective function to the model.

        Returns
        -------
        None
        """

        # Generator energy cost
        def gen_cost_ene_expr(model):
            return sum(self.nygrid.gencost_0[t, g] + self.nygrid.gencost_1[t, g] * model.PG[t, g]
                       for g in self.generators for t in self.times)

        # Penalty for violating ramp down limits
        def ramp_down_penalty_expr(model):
            return sum(model.s_ramp_down[t, g]
                       for g in self.generators for t in self.times) * self.PenaltyForRampViolation

        # Penalty for violating ramp up limits
        def ramp_up_penalty_expr(model):
            return sum(model.s_ramp_up[t, g]
                       for g in self.generators for t in self.times) * self.PenaltyForRampViolation

        # Penalty for violating interface flow limits
        def if_penalty_expr(model):
            return sum(model.s_if_max[t, n] + model.s_if_min[t, n]
                       for n in self.interfaces for t in self.times) * self.PenaltyForInterfaceMWViolation

        # Penalty for violating branch flow limits
        def br_penalty_expr(model):
            return sum(model.s_br_max[t, n] + model.s_br_min[t, n]
                       for n in self.branches for t in self.times) * self.PenaltyForBranchMwViolation

        self.model.obj = pyo.Objective(expr=(gen_cost_ene_expr(self.model)
                                             + ramp_down_penalty_expr(self.model)
                                             + ramp_up_penalty_expr(self.model)
                                             + if_penalty_expr(self.model)
                                             + br_penalty_expr(self.model)),
                                       sense=pyo.minimize)

        logging.debug('Added objective function.')

    def add_constrs_ed(self):
        """
        Add constraints for ED module.

        Returns
        -------
        None
        """

        # 1.1. Generator real power output upper limit
        def gen_power_max_rule(model, t, g):
            return model.PG[t, g] <= self.nygrid.gen_max[t, g]

        self.model.c_gen_max = pyo.Constraint(self.times, self.generators,
                                              rule=gen_power_max_rule)

        # 1.2. Generator real power output lower limit
        def gen_power_min_rule(model, t, g):
            return - model.PG[t, g] <= - self.nygrid.gen_min[t, g]

        self.model.c_gen_min = pyo.Constraint(self.times, self.generators,
                                              rule=gen_power_min_rule)

        # 2.1. Generator ramp rate downward limit
        def gen_ramp_rate_down_rule(model, t, g):
            if t == 0:
                if self.nygrid.gen_init is not None:
                    return - model.PG[t, g] + self.nygrid.gen_init[g] <= \
                        self.nygrid.ramp_down[t, g] + model.s_ramp_down[t, g]
                else:
                    return pyo.Constraint.Skip
            else:
                return - model.PG[t, g] + model.PG[t - 1, g] <= self.nygrid.ramp_down[t, g] + model.s_ramp_down[t, g]

        self.model.c_gen_ramp_down = pyo.Constraint(self.times, self.generators,
                                                    rule=gen_ramp_rate_down_rule)

        # 2.2. Generator ramp rate limit
        def gen_ramp_rate_up_rule(model, t, g):
            if t == 0:
                if self.nygrid.gen_init is not None:
                    return model.PG[t, g] - self.nygrid.gen_init[g] <= \
                        self.nygrid.ramp_up[t, g] + model.s_ramp_up[t, g]
                else:
                    return pyo.Constraint.Skip
            else:
                return model.PG[t, g] - model.PG[t - 1, g] <= self.nygrid.ramp_up[t, g] + model.s_ramp_up[t, g]

        self.model.c_gen_ramp_up = pyo.Constraint(self.times, self.generators,
                                                  rule=gen_ramp_rate_up_rule)

        # 3.1. DC line power balance constraint
        def dc_line_power_balance_rule(model, t, n):
            return model.PG[t, self.nygrid.dc_idx_f[n]] == - model.PG[t, self.nygrid.dc_idx_t[n]]

        self.model.c_dcline = pyo.Constraint(self.times, self.dclines,
                                             rule=dc_line_power_balance_rule)

        logging.debug('Added constraints for ED module.')

    def add_constrs_pf(self):
        """
        Add constraints for PF module.

        Returns
        -------
        None
        """

        # 1.1. Branch flow definition
        def branch_flow_rule(model, t, br):
            return model.PF[t, br] == sum(self.nygrid.Bf[br, b] * model.VA[t, b]
                                          for b in self.buses)

        self.model.c_br_flow = pyo.Constraint(self.times, self.branches,
                                              rule=branch_flow_rule)

        # 1.2. Branch flow upper limit
        def branch_flow_max_rule(model, t, br):
            return model.PF[t, br] <= self.nygrid.br_max[br] + model.s_br_max[t, br]

        self.model.c_br_max = pyo.Constraint(self.times, self.branches,
                                             rule=branch_flow_max_rule)

        # 1.3. Branch flow lower limit
        def branch_flow_min_rule(model, t, br):
            return - model.PF[t, br] <= - self.nygrid.br_min[br] + model.s_br_min[t, br]

        self.model.c_br_min = pyo.Constraint(self.times, self.branches,
                                             rule=branch_flow_min_rule)

        # 2.1. DC power flow constraint
        def dc_power_flow_rule(model, t, b):
            return sum(self.nygrid.gen_map[b, g] * model.PG[t, g] for g in self.generators) \
                - sum(self.nygrid.load_map[b, ld] * self.nygrid.load_pu[t, ld] for ld in self.loads) \
                == sum(self.nygrid.B[b, b_] * model.VA[t, b_] for b_ in self.buses)

        self.model.c_pf = pyo.Constraint(self.times, self.buses,
                                         rule=dc_power_flow_rule)

        # 3.1. Interface flow upper limit
        def interface_flow_max_rule(model, t, n):
            br_dir = self.nygrid.if_br_dir[n]
            br_idx = self.nygrid.if_br_idx[n]
            return sum(br_dir[i] * model.PF[t, br_idx[i]] for i in range(len(br_idx))) \
                <= self.nygrid.if_lims_max[n] + model.s_if_max[t, n]

        self.model.c_if_max = pyo.Constraint(self.times, self.interfaces,
                                             rule=interface_flow_max_rule)

        # 3.2. Interface flow lower limit
        def interface_flow_min_rule(model, t, n):
            br_dir = self.nygrid.if_br_dir[n]
            br_idx = self.nygrid.if_br_idx[n]
            return - sum(br_dir[i] * model.PF[t, br_idx[i]] for i in range(len(br_idx))) \
                <= - self.nygrid.if_lims_min[n] + model.s_if_min[t, n]

        self.model.c_if_min = pyo.Constraint(self.times, self.interfaces,
                                             rule=interface_flow_min_rule)

        logging.debug('Added constraints for PF module.')

    def add_constrs_uc(self):
        """
        Add constraints for UC module.

        Returns
        _______
        None
        """

        raise NotImplementedError('UC module is not implemented yet.')

    def add_constrs_rs(self):
        """
        Add constraints for RS module.

        Returns
        _______
        None
        """

        raise NotImplementedError('RS module is not implemented yet.')

    def add_constrs_es(self):
        """
        Add constraints for ES module.

        Returns
        _______
        None
        """

        raise NotImplementedError('ES module is not implemented yet.')

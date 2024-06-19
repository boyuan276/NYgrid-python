import pyomo.environ as pyo
import numpy as np
import logging


class Optimizer:

    def __init__(self, nygrid):
        """
        Initialize the NYGrid model.

        Parameters
        ----------
        nygrid : nygrid.nygrid.NYGrid

        Returns
        -------
        Optimizer: nygrid.optimizer.Optimizer
            OPF model.
        """

        # Define pyomo model
        self.nygrid = nygrid

        if nygrid.model is None:
            self.model = pyo.ConcreteModel(name='multi-period DC OPF')
        else:
            self.model = nygrid.model

        # Define sets
        self.times = range(self.nygrid.NT)
        self.buses = range(self.nygrid.NB)
        self.branches = range(self.nygrid.NBR)
        self.generators = range(self.nygrid.NG)
        self.loads = range(self.nygrid.NL)
        self.interfaces = range(self.nygrid.NIF)
        self.dclines = range(self.nygrid.NDCL)
        self.esrs = range(self.nygrid.NESR)

    def add_vars_ed(self):
        """
        Add variables for ED module.

        Returns
        -------
        Optimizer: nygrid.optimizer.Optimizer
            OPF model.
        """

        # Generator real power output
        self.model.PG = pyo.Var(self.times, self.generators,
                                within=pyo.Reals, initialize=1)

        # Load real power consumption
        self.model.PL = pyo.Var(self.times, self.loads,
                                within=pyo.Reals, initialize=1)

        # Slack variable for ramp rate downward limit
        self.model.s_ramp_down = pyo.Var(self.times, self.generators,
                                         within=pyo.NonNegativeReals, initialize=0)

        # Slack variable for ramp rate upward limit
        self.model.s_ramp_up = pyo.Var(self.times, self.generators,
                                       within=pyo.NonNegativeReals, initialize=0)

        # Slack variable for over generation
        self.model.s_over_gen = pyo.Var(self.times,
                                        within=pyo.NonNegativeReals, initialize=0)

        # Slack variable for load shed
        self.model.s_load_shed = pyo.Var(self.times,
                                         within=pyo.NonNegativeReals, initialize=0)

        logging.debug('Added variables for ED module.')

    def add_vars_pf(self):
        """
        Add variables for PF module.

        Returns
        -------
        None
        """

        if not self.nygrid.UsePTDF:
            # Bus phase angle to calculate DC power flow
            self.model.VA = pyo.Var(self.times, self.buses,
                                    within=pyo.Reals, initialize=0, bounds=(-2 * np.pi, 2 * np.pi))
        else:
            # Otherwise, use linearized DC power flow using PTDF
            # Power injection at each bus
            self.model.PBUS = pyo.Var(self.times, self.buses,
                                      within=pyo.Reals, initialize=0)

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
        """
        Add variables for the ES module.

        Returns
        -------
        None
        """

        # ESR real power output in charging mode
        self.model.esrPCrg = pyo.Var(self.times, self.esrs,
                                     within=pyo.NonNegativeReals, initialize=0)

        # ESR real power output in discharging mode
        self.model.esrPDis = pyo.Var(self.times, self.esrs,
                                     within=pyo.NonNegativeReals, initialize=0)

        # ESR state of charge
        self.model.esrSOC = pyo.Var(self.times, self.esrs,
                                    within=pyo.NonNegativeReals, initialize=0)

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

        # ESR energy cost
        def esr_cost_ene_expr(model):
            return sum(self.nygrid.esrcost_crg[t, esr] * model.esrPCrg[t, esr]
                       + self.nygrid.esrcost_dis[t, esr] * model.esrPDis[t, esr]
                       for esr in self.esrs for t in self.times)

        def over_gen_penalty_expr(model):
            return sum(model.s_over_gen[t] for t in self.times) * self.nygrid.PenaltyForOverGeneration

        def load_shed_penalty_expr(model):
            return sum(model.s_load_shed[t] for t in self.times) * self.nygrid.PenaltyForLoadShed

        # Penalty for violating ramp down limits
        def ramp_down_penalty_expr(model):
            return sum(model.s_ramp_down[t, g]
                       for g in self.generators for t in self.times) * self.nygrid.PenaltyForRampViolation

        # Penalty for violating ramp up limits
        def ramp_up_penalty_expr(model):
            return sum(model.s_ramp_up[t, g]
                       for g in self.generators for t in self.times) * self.nygrid.PenaltyForRampViolation

        # Penalty for violating interface flow limits
        def if_max_penalty_expr(model):
            return sum(model.s_if_max[t, n]
                       for n in self.interfaces for t in self.times) * self.nygrid.PenaltyForInterfaceMWViolation

        def if_min_penalty_expr(model):
            return sum(model.s_if_min[t, n]
                       for n in self.interfaces for t in self.times) * self.nygrid.PenaltyForInterfaceMWViolation

        # Penalty for violating branch flow limits
        def br_max_penalty_expr(model):
            return sum(model.s_br_max[t, n]
                       for n in self.branches for t in self.times) * self.nygrid.PenaltyForBranchMwViolation

        def br_min_penalty_expr(model):
            return sum(model.s_br_min[t, n]
                       for n in self.branches for t in self.times) * self.nygrid.PenaltyForBranchMwViolation

        self.model.obj = pyo.Objective(expr=(gen_cost_ene_expr(self.model)
                                             + esr_cost_ene_expr(self.model)
                                             + over_gen_penalty_expr(self.model) + load_shed_penalty_expr(self.model)
                                             + ramp_down_penalty_expr(self.model) + ramp_up_penalty_expr(self.model)
                                             + if_max_penalty_expr(self.model) + if_min_penalty_expr(self.model)
                                             + br_max_penalty_expr(self.model)) + br_min_penalty_expr(self.model),
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
            return model.PG[t, self.nygrid.dcline_idx_f[n]] == - model.PG[t, self.nygrid.dcline_idx_t[n]]

        self.model.c_dcline = pyo.Constraint(self.times, self.dclines,
                                             rule=dc_line_power_balance_rule)

        logging.debug('Added constraints for ED module.')

        # 4.1. Load real power set point
        def load_power_rule(model, t, ld):
            return model.PL[t, ld] == self.nygrid.load_pu[t, ld]

        self.model.c_load_set = pyo.Constraint(self.times, self.loads,
                                               rule=load_power_rule)

    def add_constrs_pf(self):
        """
        Add constraints for PF module.

        Returns
        -------
        None
        """

        if not self.nygrid.UsePTDF:
            # 1.1. DC power flow constraint
            def dc_power_flow_rule(model, t, b):
                return sum(self.nygrid.gen_map[b, g] * model.PG[t, g] for g in self.generators) \
                    - sum(self.nygrid.load_map[b, ld] * self.model.PL[t, ld] for ld in self.loads) \
                    == sum(self.nygrid.B[b, b_] * model.VA[t, b_] for b_ in self.buses)

            self.model.c_pf = pyo.Constraint(self.times, self.buses,
                                             rule=dc_power_flow_rule)

            # 1.2. Branch flow definition
            def branch_flow_rule(model, t, br):
                return model.PF[t, br] == sum(self.nygrid.Bf[br, b] * model.VA[t, b]
                                              for b in self.buses)

            self.model.c_br_flow = pyo.Constraint(self.times, self.branches,
                                                  rule=branch_flow_rule)

        else:
            # 1.3. System-wide energy balance constraint
            def energy_balance_rule(model, t):
                return sum(model.PG[t, g] for g in self.generators) - model.s_over_gen[t] + model.s_load_shed[t] \
                    == sum(self.model.PL[t, ld] for ld in self.loads)

            self.model.c_energy_balance = pyo.Constraint(self.times,
                                                         rule=energy_balance_rule)

            # 1.4. Power injection at each bus
            def bus_power_inj_rule(model, t, b):
                return model.PBUS[t, b] == (sum(self.nygrid.gen_map[b, g] * model.PG[t, g] for g in self.generators)
                                            - sum(self.nygrid.load_map[b, ld] * self.model.PL[t, ld]
                                                  for ld in self.loads))

            self.model.c_bus_power_inj = pyo.Constraint(self.times, self.buses,
                                                        rule=bus_power_inj_rule)

            # 1.4. Linearized DC power flow using PTDF
            def dc_power_flow_ptdf_rule(model, t, br):
                return model.PF[t, br] == sum(self.nygrid.PTDF[br, b] * model.PBUS[t, b] for b in self.buses)

            self.model.c_pf_ptdf = pyo.Constraint(self.times, self.branches,
                                                  rule=dc_power_flow_ptdf_rule)

        # 2.2. Branch flow upper limit
        def branch_flow_max_rule(model, t, br):
            return model.PF[t, br] <= self.nygrid.br_max[br] + model.s_br_max[t, br]

        self.model.c_br_max = pyo.Constraint(self.times, self.branches,
                                             rule=branch_flow_max_rule)

        # 2.3. Branch flow lower limit
        def branch_flow_min_rule(model, t, br):
            return - model.PF[t, br] <= - self.nygrid.br_min[br] + model.s_br_min[t, br]

        self.model.c_br_min = pyo.Constraint(self.times, self.branches,
                                             rule=branch_flow_min_rule)

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

        # 1.1. ESR real power output upper limit in charging mode
        def esr_power_crg_max_rule(model, t, esr):
            return model.esrPCrg[t, esr] <= self.nygrid.esr_crg_max[t, esr]

        self.model.c_esr_power_crg_max = pyo.Constraint(self.times, self.esrs,
                                                        rule=esr_power_crg_max_rule)

        # 1.2. ESR real power output upper limit in discharging mode
        def esr_power_dis_max_rule(model, t, esr):
            return model.esrPDis[t, esr] <= self.nygrid.esr_dis_max[t, esr]

        self.model.c_esr_power_dis_max = pyo.Constraint(self.times, self.esrs,
                                                        rule=esr_power_dis_max_rule)

        # 1.3. ESR combined real power output
        def esr_power_combined_rule(model, t, esr):
            return model.PG[t, self.nygrid.esr_idx[esr]] == model.esrPDis[t, esr] - model.esrPCrg[t, esr]

        self.model.c_esr_power_combined = pyo.Constraint(self.times, self.esrs,
                                                         rule=esr_power_combined_rule)

        # 2.1. ESR SOC update
        def esr_soc_update_rule(model, t, esr):
            if t == 0:
                if self.nygrid.esr_init is not None:
                    return model.esrSOC[t, esr] == self.nygrid.esr_init[esr] \
                        + model.esrPCrg[t, esr] * self.nygrid.esr_crg_eff[t, esr] \
                        - model.esrPDis[t, esr] / self.nygrid.esr_dis_eff[t, esr]
                else:
                    return pyo.Constraint.Skip
            else:
                return model.esrSOC[t, esr] == model.esrSOC[t - 1, esr] \
                    + model.esrPCrg[t, esr] * self.nygrid.esr_crg_eff[t, esr] \
                    - model.esrPDis[t, esr] / self.nygrid.esr_dis_eff[t, esr]

        self.model.c_esr_soc_update = pyo.Constraint(self.times, self.esrs,
                                                     rule=esr_soc_update_rule)

        # 2.2. ESR SOC upper limit
        def esr_soc_max_rule(model, t, esr):
            return model.esrSOC[t, esr] <= self.nygrid.esr_soc_max[t, esr]

        self.model.c_esr_soc_max = pyo.Constraint(self.times, self.esrs,
                                                  rule=esr_soc_max_rule)

        # 2.3. ESR SOC lower limit
        def esr_soc_min_rule(model, t, esr):
            return model.esrSOC[t, esr] >= self.nygrid.esr_soc_min[t, esr]

        self.model.c_esr_soc_min = pyo.Constraint(self.times, self.esrs,
                                                  rule=esr_soc_min_rule)

        # 2.4. ESR SOC target
        def esr_soc_target_rule(model, t, esr):
            if t == self.nygrid.NT - 1:
                return model.esrSOC[t, esr] == self.nygrid.esr_target[esr]
            else:
                return pyo.Constraint.Skip

        self.model.c_esr_soc_target = pyo.Constraint(self.times, self.esrs,
                                                     rule=esr_soc_target_rule)

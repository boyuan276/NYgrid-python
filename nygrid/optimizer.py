import pyomo.environ as pyo
import numpy as np
import logging

import nygrid.nygrid as ng_grid


class Optimizer:

    def __init__(self, nygrid_sim):
        """
        Initialize the NYGrid model.

        Parameters
        ----------
        nygrid_sim : nygrid.nygrid.NYGrid
            NYGrid simulation object.

        Returns
        -------
        Optimizer: nygrid.optimizer.Optimizer
            OPF model.
        """

        # Define pyomo model
        self.nygrid = nygrid_sim

        if nygrid_sim.model is None:
            self.model = pyo.ConcreteModel(name='multi-period DC OPF')
        else:
            self.model = nygrid_sim.model

        # Define sets
        self.times = range(self.nygrid.NT)
        self.buses = range(self.nygrid.NB)
        self.branches = range(self.nygrid.NBR)
        # Set of total generators
        self.generators = range(self.nygrid.NG)
        # Set of generators that need to be committed (All - offline - mustrun)
        self.generators_avail = range(self.nygrid.NG_avail)
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

    def add_vars_uc(self):
        """
        Add variables for UC module.

        Returns
        -------
        None
        """

        # Binary commitment state [1 if generator is on, 0 otherwise]
        self.model.u = pyo.Var(self.times, self.generators_avail,
                               within=pyo.Binary, initialize=0)
        
        # Binary startup states [1 if generator has a startup, 0 otherwise]
        self.model.v = pyo.Var(self.times, self.generators_avail,
                                 within=pyo.Binary, initialize=0)
        
        # Binary shutdown states [1 if generator has a shutdown, 0 otherwise]
        self.model.w = pyo.Var(self.times, self.generators_avail,
                                 within=pyo.Binary, initialize=0)

    def add_vars_pf(self):
        """
        Add variables for PF module.

        Returns
        -------
        None
        """

        if self.nygrid.UsePTDF:
            # Use linearized DC power flow using PTDF
            # Power injection at each bus
            self.model.PBUS = pyo.Var(self.times, self.buses,
                                      within=pyo.Reals, initialize=0)
        else:
            # Otherwise, Use bus phase angle to calculate DC power flow
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

        self.model.s_esr_pcrg = pyo.Var(self.times, self.esrs,
                                        within=pyo.NonNegativeReals, initialize=0)
        self.model.s_esr_pdis = pyo.Var(self.times, self.esrs,
                                        within=pyo.NonNegativeReals, initialize=0)
        self.model.s_esr_soc_min = pyo.Var(self.times, self.esrs,
                                           within=pyo.NonNegativeReals, initialize=0)
        self.model.s_esr_soc_max = pyo.Var(self.times, self.esrs,
                                           within=pyo.NonNegativeReals, initialize=0)
        self.model.s_esr_soc_overt = pyo.Var(self.times, self.esrs,
                                             within=pyo.NonNegativeReals, initialize=0)
        self.model.s_esr_soc_undert = pyo.Var(self.times, self.esrs,
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
            # return sum(self.nygrid.gencost_0[t, g] + self.nygrid.gencost_1[t, g] * model.PG[t, g]
            #            for g in self.generators for t in self.times)
            return sum(self.nygrid.gencost_1[t, g] * model.PG[t, g]
                        for g in self.generators for t in self.times)
        
        # Generator unit commitment cost
        def gen_cost_noload_expr(model):
            return sum(self.nygrid.gencost_0[t, self.nygrid.gen_idx_avail[ga]] * model.u[t, ga]
                       for ga in self.generators_avail for t in self.times)
        
        # Generator startup cost
        def gen_cost_startup_expr(model):
            return sum(self.nygrid.gencost_startup[t, self.nygrid.gen_idx_avail[ga]] * model.v[t, ga]
                       for ga in self.generators_avail for t in self.times)
        
        # Generator shutdown cost
        def gen_cost_shutdown_expr(model):
            return sum(self.nygrid.gencost_shutdown[t, self.nygrid.gen_idx_avail[ga]] * model.w[t, ga]
                       for ga in self.generators_avail for t in self.times)

        # ESR energy cost
        def esr_cost_ene_expr(model):
            return sum(- self.nygrid.esrcost_crg[t, esr] * model.esrPCrg[t, esr]
                       + self.nygrid.esrcost_dis[t, esr] * model.esrPDis[t, esr]
                       for esr in self.esrs for t in self.times)

        # Penalty for over generation at system level
        def over_gen_penalty_expr(model):
            return sum(model.s_over_gen[t] for t in self.times) * \
                self.nygrid.PenaltyForOverGeneration

        # Penalty for load shedding at system level
        def load_shed_penalty_expr(model):
            return sum(model.s_load_shed[t] for t in self.times) * \
                self.nygrid.PenaltyForLoadShed

        # Penalty for violating ramp down limits
        def ramp_down_penalty_expr(model):
            return sum(model.s_ramp_down[t, g]
                       for g in self.generators for t in self.times) * \
                        self.nygrid.PenaltyForRampViolation

        # Penalty for violating ramp up limits
        def ramp_up_penalty_expr(model):
            return sum(model.s_ramp_up[t, g]
                       for g in self.generators for t in self.times) * \
                        self.nygrid.PenaltyForRampViolation

        # Penalty for violating interface flow limits
        def if_max_penalty_expr(model):
            return sum(model.s_if_max[t, n]
                       for n in self.interfaces for t in self.times) * \
                        self.nygrid.PenaltyForInterfaceMWViolation

        def if_min_penalty_expr(model):
            return sum(model.s_if_min[t, n]
                       for n in self.interfaces for t in self.times) * \
                        self.nygrid.PenaltyForInterfaceMWViolation

        # Penalty for violating branch flow upper limits
        def br_max_penalty_expr(model):
            return sum(model.s_br_max[t, n]
                       for n in self.branches for t in self.times) * \
                        self.nygrid.PenaltyForBranchMwViolation

        # Penalty for violating branch flow lower limits
        def br_min_penalty_expr(model):
            return sum(model.s_br_min[t, n]
                       for n in self.branches for t in self.times) * \
                        self.nygrid.PenaltyForBranchMwViolation

        # Penalty for violating ESR charging power limits
        def esr_pcrg_penalty_expr(model):
            return sum(model.s_esr_pcrg[t, esr]
                       for esr in self.esrs for t in self.times) * \
                        self.nygrid.PenaltyForESRPowerViolation

        # Penalty for violating ESR discharging power limits
        def esr_pdis_penalty_expr(model):
            return sum(model.s_esr_pdis[t, esr]
                       for esr in self.esrs for t in self.times) * \
                        self.nygrid.PenaltyForESRPowerViolation

        # Penalty for violating ESR SOC upper limits
        def esr_soc_max_penalty_expr(model):
            return sum(model.s_esr_soc_max[t, esr]
                       for esr in self.esrs for t in self.times) * \
                        self.nygrid.PenaltyForESRSOCLimitViolation

        # Penalty for violating ESR SOC lower limits
        def esr_soc_min_penalty_expr(model):
            return sum(model.s_esr_soc_min[t, esr]
                       for esr in self.esrs for t in self.times) * \
                        self.nygrid.PenaltyForESRSOCLimitViolation

        # Penalty for terminal SOC greater than target
        def esr_soc_overt_penalty_expr(model):
            return sum(model.s_esr_soc_overt[t, esr]
                       for esr in self.esrs for t in self.times) * \
                        self.nygrid.PenaltyForESRSOCTargetViolation

        # Penalty for terminal SOC less than target
        def esr_soc_undert_penalty_expr(model):
            return sum(model.s_esr_soc_undert[t, esr]
                       for esr in self.esrs for t in self.times) * \
                        self.nygrid.PenaltyForESRSOCTargetViolation

        # Objective function
        self.model.obj = pyo.Objective(expr=(
            gen_cost_ene_expr(self.model)
            + gen_cost_noload_expr(self.model)
            + gen_cost_startup_expr(self.model)
            + gen_cost_shutdown_expr(self.model)
            + esr_cost_ene_expr(self.model)
            + over_gen_penalty_expr(self.model) 
            + load_shed_penalty_expr(self.model)
            + ramp_down_penalty_expr(self.model) 
            + ramp_up_penalty_expr(self.model)
            + if_max_penalty_expr(self.model) 
            + if_min_penalty_expr(self.model)
            + br_max_penalty_expr(self.model) 
            + br_min_penalty_expr(self.model)
            + esr_pcrg_penalty_expr(self.model) 
            + esr_pdis_penalty_expr(self.model)
            + esr_soc_max_penalty_expr(self.model) 
            + esr_soc_min_penalty_expr(self.model)
            + esr_soc_overt_penalty_expr(self.model)
            + esr_soc_undert_penalty_expr(self.model)
        ), sense=pyo.minimize)

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
                return - model.PG[t, g] + model.PG[t - 1, g] <= \
                    self.nygrid.ramp_down[t, g] + model.s_ramp_down[t, g]

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
                return model.PG[t, g] - model.PG[t - 1, g] <= \
                    self.nygrid.ramp_up[t, g] + model.s_ramp_up[t, g]

        self.model.c_gen_ramp_up = pyo.Constraint(self.times, self.generators,
                                                  rule=gen_ramp_rate_up_rule)

        # 3.1. DC line power balance constraint
        def dc_line_power_balance_rule(model, t, n):
            return (model.PG[t, self.nygrid.dcline_idx_f[n]] == \
                    - model.PG[t, self.nygrid.dcline_idx_t[n]])

        self.model.c_dcline = pyo.Constraint(self.times, self.dclines,
                                             rule=dc_line_power_balance_rule)

        logging.debug('Added constraints for ED module.')

        # 4.1. Load real power set point
        def load_power_rule(model, t, ld):
            return model.PL[t, ld] == self.nygrid.load_pu[t, ld]

        self.model.c_load_set = pyo.Constraint(self.times, self.loads,
                                               rule=load_power_rule)

    def add_constrs_uc(self):
        """
        Add constraints for UC module.

        Returns
        _______
        None
        """

        # 1.1. Generator real power output upper limit with commitment status
        def gen_power_max_cmt_rule(model, t, ga):
            g = self.nygrid.gen_idx_avail[ga]
            return model.PG[t, g] <= self.nygrid.gen_max[t, g] * model.u[t, ga]
        
        self.model.c_gen_max_cmt = pyo.Constraint(self.times, self.generators_avail,
                                                rule=gen_power_max_cmt_rule)
        
        # 1.2. Generator real power output lower limit with commitment status
        def gen_power_min_cmt_rule(model, t, ga):
            g = self.nygrid.gen_idx_avail[ga]
            return model.PG[t, g] >= self.nygrid.gen_min[t, g] * model.u[t, ga]
        
        self.model.c_gen_min_cmt = pyo.Constraint(self.times, self.generators_avail,
                                                rule=gen_power_min_cmt_rule)
        
        # 2.1. Generator commitment status
        def gen_commit_rule(model, t, ga):
            g = self.nygrid.gen_idx_avail[ga]
            if t == 0:
                if self.nygrid.gen_init_cmt is not None:
                    return model.u[t, ga] == self.nygrid.gen_init_cmt[ga] \
                        + model.v[t, ga] - model.w[t, ga]
                else:
                    return pyo.Constraint.Skip
            else:
                return model.u[t, ga] == model.u[t - 1, ga] \
                    + model.v[t, ga] - model.w[t, ga]

        self.model.c_gen_commitment = pyo.Constraint(self.times, self.generators_avail,
                                                    rule=gen_commit_rule)
        
        def gen_commit_rule_2(model, t, ga):
            return model.v[t, ga] + model.w[t, ga] <= 1
        
        self.model.c_gen_commitment_2 = pyo.Constraint(self.times, self.generators_avail,
                                                    rule=gen_commit_rule_2)

        # 2.2. Generator minimum down time constraint
        def gen_min_up_time_rule(model, t, ga):
            g = self.nygrid.gen_idx_avail[ga]
            if t < self.nygrid.min_up_time[g]:
                return pyo.Constraint.Skip
            else:
                startup_count = 0
                for time in range(self.nygrid.min_up_time[g]):
                    startup_count += model.v[t - time, ga]
                return startup_count <= model.u[t, ga]

        self.model.c_gen_min_up_time = pyo.Constraint(self.times, self.generators_avail,
                                                        rule=gen_min_up_time_rule)

        # 2.3. Generator minimum up time constraint
        def gen_min_down_time_rule(model, t, ga):
            g = self.nygrid.gen_idx_avail[ga]
            if t < self.nygrid.min_down_time[g]:
                return pyo.Constraint.Skip
            else:
                shutdown_count = 0
                for time in range(self.nygrid.min_down_time[g]):
                    shutdown_count += model.w[t - time, ga]
                return shutdown_count <= 1- model.u[t, ga]

        self.model.c_gen_min_down_time = pyo.Constraint(self.times, self.generators_avail,
                                                      rule=gen_min_down_time_rule)   
    
    def add_constrs_pf(self):
        """
        Add constraints for PF module.

        Returns
        -------
        None
        """

        if self.nygrid.UsePTDF:
            # 1.1a. System-wide energy balance constraint
            def energy_balance_rule(model, t):
                return sum(model.PG[t, g] for g in self.generators) \
                    - model.s_over_gen[t] + model.s_load_shed[t] \
                    == sum(model.PL[t, ld] for ld in self.loads)

            self.model.c_energy_balance = pyo.Constraint(self.times,
                                                         rule=energy_balance_rule)

            # 1.2a. Power injection at each bus
            def bus_power_inj_rule(model, t, b):
                return model.PBUS[t, b] == (sum(self.nygrid.gen_map[b, g] * model.PG[t, g]
                                                for g in self.generators)
                                            - sum(self.nygrid.load_map[b, ld] * model.PL[t, ld]
                                                  for ld in self.loads))

            self.model.c_bus_power_inj = pyo.Constraint(self.times, self.buses,
                                                        rule=bus_power_inj_rule)

            # 1.3a. Linearized DC power flow using PTDF
            def dc_power_flow_ptdf_rule(model, t, br):
                return model.PF[t, br] == sum(self.nygrid.PTDF[br, b] * \
                                              model.PBUS[t, b] for b in self.buses)

            self.model.c_pf_ptdf = pyo.Constraint(self.times, self.branches,
                                                  rule=dc_power_flow_ptdf_rule)

        else:
            # 1.1b. DC power flow constraint
            def dc_power_flow_rule(model, t, b):
                return sum(self.nygrid.gen_map[b, g] * model.PG[t, g] for g in self.generators) \
                    - sum(self.nygrid.load_map[b, ld] * model.PL[t, ld] for ld in self.loads) \
                    == sum(self.nygrid.B[b, b_] * model.VA[t, b_] for b_ in self.buses)

            self.model.c_pf = pyo.Constraint(self.times, self.buses,
                                             rule=dc_power_flow_rule)

            # 1.2b. Branch flow definition
            def branch_flow_rule(model, t, br):
                return model.PF[t, br] == sum(self.nygrid.Bf[br, b] * model.VA[t, b]
                                              for b in self.buses)

            self.model.c_br_flow = pyo.Constraint(self.times, self.branches,
                                                  rule=branch_flow_rule)

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
            return model.esrPCrg[t, esr] <= \
                self.nygrid.esr_crg_max[t, esr] + model.s_esr_pcrg[t, esr]

        self.model.c_esr_power_crg_max = pyo.Constraint(self.times, self.esrs,
                                                        rule=esr_power_crg_max_rule)

        # 1.2. ESR real power output upper limit in discharging mode
        def esr_power_dis_max_rule(model, t, esr):
            return model.esrPDis[t, esr] <= \
                self.nygrid.esr_dis_max[t, esr] + model.s_esr_pdis[t, esr]

        self.model.c_esr_power_dis_max = pyo.Constraint(self.times, self.esrs,
                                                        rule=esr_power_dis_max_rule)

        # 1.3. ESR combined real power output
        def esr_power_combined_rule(model, t, esr):
            return model.PG[t, self.nygrid.esr_idx[esr]] == \
                model.esrPDis[t, esr] - model.esrPCrg[t, esr]

        self.model.c_esr_power_combined = pyo.Constraint(self.times, self.esrs,
                                                         rule=esr_power_combined_rule)

        # 2.1. ESR SOC update
        def esr_soc_update_rule(model, t, esr):
            if t == 0:
                if self.nygrid.esr_init is not None:
                    return model.esrSOC[t, esr] == self.nygrid.esr_init[esr] \
                        + model.esrPCrg[t, esr] * self.nygrid.esr_crg_eff[t, esr] \
                        - model.esrPDis[t, esr] / \
                        self.nygrid.esr_dis_eff[t, esr]
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
            return model.esrSOC[t, esr] <= self.nygrid.esr_soc_max[t, esr] + model.s_esr_soc_max[t, esr]

        self.model.c_esr_soc_max = pyo.Constraint(self.times, self.esrs,
                                                  rule=esr_soc_max_rule)

        # 2.3. ESR SOC lower limit
        def esr_soc_min_rule(model, t, esr):
            return - model.esrSOC[t, esr] <= - self.nygrid.esr_soc_min[t, esr] + model.s_esr_soc_min[t, esr]

        self.model.c_esr_soc_min = pyo.Constraint(self.times, self.esrs,
                                                  rule=esr_soc_min_rule)

        # 2.4. ESR SOC target
        def esr_soc_target_rule(model, t, esr):
            if t == self.nygrid.NT - 1:
                return model.esrSOC[t, esr] == self.nygrid.esr_target[esr] \
                    + model.s_esr_soc_overt[t, esr] \
                    - model.s_esr_soc_undert[t, esr]
            else:
                return pyo.Constraint.Skip

        self.model.c_esr_soc_target = pyo.Constraint(self.times, self.esrs,
                                                     rule=esr_soc_target_rule)

"""
Class for running the NYGrid model.

Known Issues/Wishlist:
1. Better dc line model
2. Better documentation
3. Check dim of start/end datetime and load profile

"""

from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import pypower.api as pp
# Import pypower data indexing
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
from . import utils

class NYGrid:
    """
    Class for running the NYGrid model

    Parameters
    ----------
    :param 


    """
    def __init__(self, ppc, start_datetime, end_datetime, 
                 setup_yaml='dirpaths.yml', verbose=False):
        self.ppc = ppc
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.verbose = verbose

        # Format the forecast start/end and determine the total time.
        self.start_datetime = utils.format_date(start_datetime)
        self.end_datetime = utils.format_date(end_datetime)
        self.delt = self.end_datetime - self.start_datetime
        if self.verbose:
            print('Initializing NYGrid run...')
            print(f'NYGrid run starting on: {self.start_datetime}')
            print(f'NYGrid run ending on: {self.end_datetime}')

        self.timestamp_list = pd.date_range(self.start_datetime, self.end_datetime, freq='1H')

    def create_single_opf(self):
        '''
        Single-period OPF problem.

        Parameters:
            A tuple of OPF network parameters and constraints.
        
        Returns:
            model (pyomo.core.base.PyomoModel.ConcreteModel): Pyomo model for single-period OPF problem.
        '''

        model = ConcreteModel(name='single-period OPF')

        # Define variables
        model.PG = Var(range(self.NG), within=Reals,initialize=1)  # Real power generation
        model.Va = Var(range(self.NB), within=Reals, initialize=0,
                    bounds=(-2*np.pi, 2*np.pi))  # Bus voltage angle
        model.PF = Var(range(self.NBR), within=Reals, initialize=0) # Branch power flow
        model.dual = Suffix(direction=Suffix.IMPORT) # Dual variables for price information

        # Define constraints
        model.c_br_flow = ConstraintList()
        model.c_br_max = ConstraintList()
        model.c_br_min = ConstraintList()
        model.c_gen_max = ConstraintList()
        model.c_gen_min = ConstraintList()
        model.c_pf = ConstraintList()
        model.c_dcline = ConstraintList()
        model.c_if_max = ConstraintList()
        model.c_if_min = ConstraintList()

        # Line flow limit constraints
        for br in range(self.NBR):
            model.c_br_flow.add(model.PF[br] == sum(self.Bf[br, b]*model.Va[b] 
                                                    for b in range(self.NB)))
            model.c_br_max.add(model.PF[br] <= self.br_max[br])
            model.c_br_min.add(model.PF[br] >= self.br_min[br])
        
        # Generation capacity limit
        for g in range(self.NG):
            model.c_gen_max.add(model.PG[g] <= self.gen_max[g])
            model.c_gen_min.add(model.PG[g] >= self.gen_min[g])
            
        # DC power flow constraint
        for b in range(self.NB):
            model.c_pf.add(sum(self.gen_map[b, g]*model.PG[g] for g in range(self.NG)) 
                           - sum(self.load_map[b, l]*self.load_pu[l] for l in range(self.NL))
                            == sum(self.B[b, b_]*model.Va[b_] for b_ in range(self.NB)))
        
        # DC line power balance constraint
        for idx_f, idx_t in zip(self.dc_idx_f, self.dc_idx_t):
            model.c_dcline.add(model.PG[idx_f] == -model.PG[idx_t])
        
        # Interface flow limits
        for n in range(len(self.if_lims)):
            if_id, if_lims_min, if_lims_max = self.if_lims[n, :]
            br_dir_idx = self.if_map[(self.if_map[:,0] == int(if_id)), 1]
            br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
            model.c_if_max.add(if_lims_max >= sum(br_dir[i]*model.PF[br_idx[i]] 
                                                  for i in range(len(br_idx))))
            model.c_if_min.add(if_lims_min <= sum(br_dir[i]*model.PF[br_idx[i]] 
                                                  for i in range(len(br_idx))))

        def cost(model, gencost_0, gencost_1):

            cost = sum(gencost_0[i] for i in range(self.NG)) \
                + sum(gencost_1[i]*model.PG[i] for i in range(self.NG))
            return cost

        model.obj = Objective(expr=cost(model, self.gencost_0, self.gencost_1), 
                              sense=minimize)

        return model

    def create_multi_opf(self):
        '''
        Multi-period OPF problem.

        Parameters:
            A tuple of OPF network parameters and constraints.
        
        Returns:
            model (pyomo.core.base.PyomoModel.ConcreteModel): Pyomo model for multi-period OPF problem.
        '''

        model = ConcreteModel(name='multi-period OPF')

        # Define variables
        model.PG = Var(range(self.NT), range(self.NG), 
                       within=Reals, initialize=1)  # Real power generation
        model.Va = Var(range(self.NT), range(self.NB), 
                       within=Reals, initialize=0,
                    bounds=(-2*np.pi, 2*np.pi))  # Bus phase angle
        model.PF = Var(range(self.NT), range(self.NBR), within=Reals, initialize=0) # Branch power flow
        model.dual = Suffix(direction=Suffix.IMPORT) # Dual variables for price information

        # Define constraints
        model.c_br_flow = ConstraintList()
        model.c_br_max = ConstraintList()
        model.c_br_min = ConstraintList()
        model.c_gen_max = ConstraintList()
        model.c_gen_min = ConstraintList()
        model.c_pf = ConstraintList()
        model.c_gen_ramp_up = ConstraintList()
        model.c_gen_ramp_down = ConstraintList()
        model.c_dcline = ConstraintList()
        model.c_if_max = ConstraintList()
        model.c_if_min = ConstraintList()

        for t in range(self.NT):
            # Line flow limit constraints
            for br in range(self.NBR):
                model.c_br_flow.add(model.PF[t, br] == sum(self.Bf[br, b]*model.Va[t, b] 
                                                           for b in range(self.NB)))
                model.c_br_max.add(model.PF[t, br] <= self.br_max[br])
                model.c_br_min.add(model.PF[t, br] >= self.br_min[br])

            # Generation capacity limit
            for g in range(self.NG):
                model.c_gen_max.add(model.PG[t, g] <= self.gen_max[g])
                model.c_gen_min.add(model.PG[t, g] >= self.gen_min[g])

            # DC Power flow constraint
            for b in range(self.NB):
                model.c_pf.add(sum(self.gen_map[b, g]*model.PG[t, g] for g in range(self.NG)) 
                               - sum(self.load_map[b, l]*self.load_pu[t, l] for l in range(NL))
                                == sum(self.B[b, b_]*model.Va[t, b_] for b_ in range(self.NB)))
            
            # DC line power balance constraint
            for idx_f, idx_t in zip(self.dc_idx_f, self.dc_idx_t):
                model.c_dcline.add(model.PG[t, idx_f] == -model.PG[t, idx_t])

            # Interface flow limits
            for n in range(len(self.if_lims)):
                if_id, if_lims_min, if_lims_max = self.if_lims[n, :]
                br_dir_idx = self.if_map[(self.if_map[:,0] == int(if_id)), 1]
                br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
                model.c_if_min.add(if_lims_min <= sum(br_dir[i]*model.PF[t, br_idx[i]] 
                                                      for i in range(len(br_idx))))
                model.c_if_max.add(if_lims_max >= sum(br_dir[i]*model.PF[t, br_idx[i]] 
                                                      for i in range(len(br_idx))))

        for t in range(self.NT-1):
            # Ramp rate limit
            for g in range(self.NG):
                model.c_gen_ramp_down.add(-model.PG[t+1, g] + model.PG[t, g] <= self.ramp_down[g])
                model.c_gen_ramp_up.add(model.PG[t+1, g] - model.PG[t, g] <= self.ramp_up[g])

        def cost(model, gencost_0, gencost_1):
            cost = 0
            for t in range(self.NT):
                cost += sum(gencost_0[g] for g in range(self.NG)) \
                    + sum(gencost_1[g]*model.PG[t, g] for g in range(self.NG))
            return cost

        model.obj = Objective(expr=cost(model, self.gencost_0, self.gencost_1), sense=minimize)

        return model

    def get_load_data(self, load):
        self.load = load
        self.NT = self.load.shape[0]

    def process_ppc(self):
        '''
        Process PyPower case to get constraint value for the OPF problem.

        Parameters
        ----------
            ppc (dict): PyPower case in python dictionary.
            load (numpy.ndarray): A 2-d array of load at each timestep at each bus
        
        Returns
        -------
            Parameters of the network and constraints.
        '''

        # Remove user functions
        del self.ppc['userfcn']

        # Format genfuel and bus_name strings
        self.ppc['genfuel'] = np.array([str(x[0][0]) for x in self.ppc['genfuel']])
        self.ppc['bus_name'] = np.array([str(x[0][0]) for x in self.ppc['bus_name']])

        # Format interface limit data
        self.ppc['if'] = {
            'map': self.ppc['if'][0][0][0],
            'lims': self.ppc['if'][0][0][1]
        }
        self.ppc = pp.toggle_iflims(self.ppc, 'on')

        # Convert baseMVA to float
        self.ppc['baseMVA'] = float(self.ppc['baseMVA'])

        # Convert DC line to generators and add to gen matrix
        self.ppc_dc, self.num_dcline = self.convert_dcline_2_gen()

        # Convert to internal indexing
        self.ppc_int = pp.ext2int(self.ppc_dc)

        self.baseMVA = self.ppc_int['baseMVA']
        self.bus = self.ppc_int['bus']
        self.gen = self.ppc_int['gen']
        self.branch = self.ppc_int['branch']
        self.gencost = self.ppc_int['gencost']

        # Generator info
        self.gen_bus = self.gen[:, GEN_BUS].astype(int)  # what buses are they at?

        # build B matrices and phase shift injections
        B, Bf, _, _ = pp.makeBdc(self.baseMVA, self.bus, self.branch)
        self.B = B.todense()
        self.Bf = Bf.todense()

        # Problem dimensions
        self.NG = self.gen.shape[0]  # Number of generators
        self.NB = self.bus.shape[0]  # Number of buses
        self.NBR = self.branch.shape[0]  # Number of lines
        self.NL = np.count_nonzero(self.bus[:, PD])  # Number of loads

        # Get mapping from generator to bus
        self.gen_map = np.zeros((self.NB, self.NG))
        self.gen_map[self.gen_bus, range(self.NG)] = 1

        # Get index of DC line converted generators in internal indexing
        gen_i2e = self.ppc_int['order']['gen']['i2e']
        self.dc_idx_f = gen_i2e[self.NG-self.num_dcline*2: self.NG-self.num_dcline]
        self.dc_idx_t = gen_i2e[self.NG-self.num_dcline: self.NG]

        # Get mapping from load to bus
        self.load_map = np.zeros((self.NB, self.NL))
        self.load_bus = np.nonzero(self.bus[:, PD])[0]
        for i in range(len(self.load_bus)):
            self.load_map[self.load_bus[i], i] = 1

        # Generator capacity limit in p.u.
        self.gen_max = self.gen[:, PMAX]/self.baseMVA
        self.gen_min = self.gen[:, PMIN]/self.baseMVA

        # Generator upward and downward ramp limit in p.u.
        self.ramp_up = self.gen[:, RAMP_30]*2/self.baseMVA
        self.ramp_down = np.min([self.gen_max, self.ramp_up], axis=0)

        # Line flow limit in p.u.
        self.br_max = self.branch[:, RATE_A]/self.baseMVA
        # Replace default value 0 to 999
        self.br_max[self.br_max == 0] = 999.99
        self.br_min = - self.br_max

        # Linear cost coefficients in p.u.
        self.gencost_1 = self.gencost[:, COST]*self.baseMVA
        self.gencost_0 = self.gencost[:, COST+1]

        # Get interface limit information
        self.if_map = self.ppc_int['if']['map']
        self.if_lims = self.ppc_int['if']['lims']
        self.if_lims[:,1:] = self.if_lims[:,1:]/self.baseMVA
        br_dir, br_idx = np.sign(self.if_map[:,1]), np.abs(self.if_map[:,1]).astype(int)
        self.if_map[:, 1] = br_dir*(br_idx-1)
        
        # Convert load to p.u.
        self.load_pu = self.load/self.baseMVA

    def check_status(self, results):
        '''
        Check the status of a Pyomo model.

        Parameters:
            results (pyomo.opt.results.results_.SolverResults): Pyomo model results.
        
        Returns:
            status (bool): True if the model is solved successfully.
        '''

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            status = True
            print("The problem is feasible and optimal!")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            status = False
            print("The problem is infeasible!")
        else:
            status = False
            print("Something else is wrong!")
            print (str(results.solver))
        return status
    
    def convert_dcline_2_gen(self):
        '''
        Convert DC lines to generators and add their parameters in the PyPower matrices.

        Parameters:
            ppc (dict): PyPower case dictionary.
        
        Returns:
            ppc (dict): updated PyPower case dictionary.
            num_dcline (float): number of DC lines.
        '''

        # Define dcline matrix indices
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

        # Get PyPower case information
        ppc_dc = self.ppc
        baseMVA = ppc_dc['baseMVA']
        gen = ppc_dc['gen']
        gencost = ppc_dc['gencost']
        dcline = ppc_dc['dcline']
        genfuel = ppc_dc['genfuel']

        # Set gen parameters of the DC line converted generators
        num_dcline = dcline.shape[0]
        dcline_gen = np.zeros((self.num_dcline*2, 21))
        dcline_gen[:, GEN_BUS] = np.concatenate([dcline[:, DC_F_BUS], 
                                                    dcline[:, DC_T_BUS]])
        dcline_gen[:, PG] = np.concatenate([-dcline[:, DC_PF],
                                            dcline[:, DC_PF]])
        dcline_gen[:, QG] = np.concatenate([-dcline[:, DC_QF],
                                            dcline[:, DC_QF]])
        dcline_gen[:, QMAX] = np.concatenate([dcline[:, DC_QMAXF],
                                            dcline[:, DC_QMAXT]])
        dcline_gen[:, QMIN] = np.concatenate([dcline[:, DC_QMINF],
                                            dcline[:, DC_QMINT]])
        dcline_gen[:, VG] = np.concatenate([dcline[:, DC_VF],
                                            dcline[:, DC_VT]])
        dcline_gen[:, MBASE] = np.ones(num_dcline*2)*baseMVA
        dcline_gen[:, GEN_STATUS] = np.concatenate([dcline[:, DC_BR_STATUS],
                                                    dcline[:, DC_BR_STATUS]])
        # dcline_gen[:, PMAX] = np.concatenate([dcline[:, DC_PMAX],
        #                                     dcline[:, DC_PMAX]])
        dcline_gen[:, PMAX] = np.full(num_dcline*2, np.inf)
        dcline_gen[:, PMIN] = np.concatenate([dcline[:, DC_PMIN],
                                            dcline[:, DC_PMIN]])
        # Add the DC line converted generators to the gen matrix
        ppc_dc['gen'] = np.concatenate([gen, dcline_gen])

        # Set gencost parameters of the DC line converted generators
        dcline_gencost = np.zeros((num_dcline*2, 6))
        dcline_gencost[:, MODEL] = np.ones(num_dcline*2)*POLYNOMIAL
        dcline_gencost[:, NCOST] = np.ones(num_dcline*2)*2
        # Add the DC line converted generators to the gencost matrix
        ppc_dc['gencost'] = np.concatenate([gencost, dcline_gencost])

        # Add the DC line converted generators to the genfuel list
        dcline_genfuel = ['dc line']*num_dcline*2
        ppc_dc['genfuel'] = np.concatenate([genfuel, dcline_genfuel])

        return ppc_dc, num_dcline

    def check_input_dim(self):
        # Check dimensions of the input data
        if (self.gen_min.shape != self.gen_max.shape) \
            or (self.ramp_down.shape != self.ramp_up.shape):
            raise ValueError('Found mismatch in generator constraint dimensions!')
            
        if (self.br_min.shape != self.br_max.shape):
            raise ValueError('Found mismatch in branch flow limit array dimensions!')
        
    def get_results_single_opf(self, model_single_opf):
        '''
        Get results for a single-period OPF problem.

        Parameters:
            model_multi (Pyomo model): Pyomo model of single-period OPF problem.
            ppc_int (dict): a dict of PyPower case with internal indexing.
        
        Returns:
            results (dict): a dict of pandas Series, including:
                1. Generator power generation.
                2. Bus phase angle.
                3. Branch power flow.
                4. Interface flow.
                5. Bus locational marginal price (LMP).
        '''
        # Power generation
        results_pg = np.array(model_single_opf.PG[:]())*self.baseMVA
        gen_order = self.ppc_int['order']['gen']['e2i']
        results_pg = pd.Series(results_pg, index=gen_order).sort_index()
        
        # Bus phase angle
        results_va = np.array(model_single_opf.Va[:]())*180/np.pi
        # Just to compare with PyPower
        results_va = results_va - 73.4282
        # Convert negative numbers to 0-360
        results_va = np.where(results_va < 0, results_va+360, results_va)
        results_va = pd.Series(results_va)

        # Branch power flow
        branch_pf = np.array(model_single_opf.PF[:]())*self.baseMVA
        results_pf = pd.Series(branch_pf)

        # Interface flow
        if_flow = np.zeros(len(self.if_lims))
        for n in range(len(self.if_lims)):
            if_id = self.if_lims[n, 0]
            br_dir_idx = self.if_map[(self.if_map[:,0] == int(if_id)), 1]
            br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
            if_flow[n] = sum(br_dir[i]*branch_pf[br_idx[i]-1] for i in range(len(br_idx)))
            if_flow[n] = sum(br_dir[i]*branch_pf[br_idx[i]-1] for i in range(len(br_idx)))
        results_if = pd.Series(if_flow)

        bus_lmp = np.zeros(self.NB)
        for i in range(self.NB):
            bus_lmp[i] = np.abs(model_single_opf.dual[model_single_opf.c_pf[i+1]])/self.baseMVA
        results_lmp = pd.Series(bus_lmp)

        results_single_opf = {
            'PG': results_pg,
            'VA': results_va,
            'PF': results_pf,
            'IF': results_if,
            'LMP': results_lmp
        }

        return results_single_opf

    def get_results_multi_opf(self, model_multi_opf):
        '''
        Get results for a multi-period OPF problem.

        Parameters:
            model_multi (Pyomo model): Pyomo model of multi-period OPF problem.
            ppc_int (dict): a dict of PyPower case with internal indexing.
            timestamp_list (list): a list of timestamps.
        
        Returns:
            results (dict): a dict of pandas DataFrames, including:
                1. Generator power generation.
                2. Bus phase angle.
                3. Branch power flow.
                4. Interface flow.
                5. Bus locational marginal price (LMP).
        '''
        # Power generation
        results_pg = np.array(model_multi_opf.PG[:,:]()).reshape(self.NT, self.NG)*self.baseMVA
        gen_order = self.ppc_int['order']['gen']['e2i']
        results_pg = pd.DataFrame(results_pg, index=self.timestamp_list,
                                    columns=gen_order).sort_index(axis=1)
        
        # Bus phase angle
        results_va = np.array(model_multi_opf.Va[:,:]()).reshape(self.NT, self.NB)*180/np.pi
        # Just to compare with PyPower
        results_va = results_va - 73.4282
        # Convert negative numbers to 0-360
        results_va = np.where(results_va < 0, results_va+360, results_va)
        results_va = pd.DataFrame(results_va, index=self.timestamp_list)

        # Branch power flow
        branch_pf = np.array(model_multi_opf.PF[:,:]()).reshape(self.NT, self.NBR)*self.baseMVA
        results_pf = pd.DataFrame(branch_pf, index=self.timestamp_list)

        # Interface flow
        if_flow = np.zeros((self.NT, len(self.if_lims)))
        for t in range(self.NT):
            for n in range(len(self.if_lims)):
                if_id = self.if_lims[n, 0]
                br_dir_idx = self.if_map[(self.if_map[:,0] == int(if_id)), 1]
                br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
                if_flow[t, n] = sum(br_dir[i]*branch_pf[t, br_idx[i]-1] for i in range(len(br_idx)))
                if_flow[t, n] = sum(br_dir[i]*branch_pf[t, br_idx[i]-1] for i in range(len(br_idx)))
        results_if = pd.DataFrame(if_flow, index=self.timestamp_list)

        bus_lmp = np.zeros(self.NT*self.NB)
        for i in range(self.NT*self.NB):
            bus_lmp[i] = np.abs(model_multi_opf.dual[model_multi_opf.c_pf[i+1]])/self.baseMVA
        results_lmp = bus_lmp.reshape(self.num_time, self.num_bus)
        results_lmp = pd.DataFrame(results_lmp, index=self.timestamp_list)

        results = {
            'PG': results_pg,
            'VA': results_va,
            'PF': results_pf,
            'IF': results_if,
            'LMP': results_lmp
        }

        return results


"""
Class for running the NYGrid model.

Known Issues/Wishlist:
1. Better dc line model
2. Better documentation
3. Check dim of start/end datetime and load profile

"""

from pyomo.environ import *
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

class NYGrid:
    """
    Class for running the NYGrid model

    Parameters
    ----------
    :param 


    """
    def __init__(self, ppc_filename, start_datetime, end_datetime, 
                 verbose=False):
        self.ppc = pp.loadcase(ppc_filename)
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
            print(f'NYGrid run duration: {self.delt}')

        self.timestamp_list = pd.date_range(self.start_datetime, self.end_datetime, freq='1H')
        self.NT = len(self.timestamp_list)

        # User-defined parameters
        self.load_profile = None
        self.gen_profile = None
        self.genmax_profile = None
        self.genmin_profile = None
        self.genramp_profile = None
        self.gencost0_profile = None
        self.gencost1_profile = None

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
        print('Creating multi-period OPF problem ...')

        if self.verbose:
            timer = TicTocTimer()
            timer.tic('Starting timer ...')

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
                model.c_gen_max.add(model.PG[t, g] <= self.gen_max[t, g])
                model.c_gen_min.add(model.PG[t, g] >= self.gen_min[t, g])

            # DC Power flow constraint
            for b in range(self.NB):
                model.c_pf.add(sum(self.gen_map[b, g]*model.PG[t, g] for g in range(self.NG)) 
                               - sum(self.load_map[b, l]*self.load_pu[t, l] for l in range(self.NL))
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

            if t == 0:
                if self.gen_init is not None:
                    # Ramp rate limit from initial condition
                    for g in range(self.NG):
                        model.c_gen_ramp_down.add(-model.PG[t, g] + self.gen_init[g] <= self.ramp_down[t, g])
                        model.c_gen_ramp_up.add(model.PG[t, g] - self.gen_init[g] <= self.ramp_up[t, g])
            else:
                # Ramp rate limit
                for g in range(self.NG):
                    model.c_gen_ramp_down.add(-model.PG[t, g] + model.PG[t-1, g] <= self.ramp_down[t, g])
                    model.c_gen_ramp_up.add(model.PG[t, g] - model.PG[t-1, g] <= self.ramp_up[t, g])
            
            if self.verbose:
                timer.toc(f'Created constraints for time step {t} ...')

        def cost(model, gencost_0, gencost_1):
            cost = 0
            for t in range(self.NT):
                cost += sum(gencost_0[t, g] for g in range(self.NG)) \
                    + sum(gencost_1[t, g]*model.PG[t, g] for g in range(self.NG))
            return cost

        model.obj = Objective(expr=cost(model, self.gencost_0, self.gencost_1), sense=minimize)

        print('Created model ...')

        return model

    def get_load_data(self, load_profile):
        '''
        Get load data from load profile.

        Parameters
        ----------
            load_profile (str): Path to load profile csv file.
            
        Returns
        -------
            load (numpy.ndarray): A 2-d array of load at each timestamp at each bus
        '''
        self.load_profile = load_profile[self.start_datetime:self.end_datetime].to_numpy()

    def get_gen_data(self, gen_profile):
        '''
        Get generation data from generation profile.

        Parameters
        ----------
            gen_profile (str): Path to generation profile csv file.
            
        Returns
        -------
            gen (numpy.ndarray): A 2-d array of generation at each timestamp at each bus
        '''
        self.gen_profile = gen_profile[self.start_datetime:self.end_datetime].to_numpy()

    def get_genmax_data(self, genmax_profile):
        '''
        Get generation capacity data from generation capacity profile.

        Parameters
        ----------
            genmax_profile (str): Path to generation capacity profile csv file.
            
        Returns
        -------
            gen_max (numpy.ndarray): A 2-d array of generation capacity at each bus
        '''
        self.genmax_profile = genmax_profile[self.start_datetime:self.end_datetime].to_numpy()

    def get_genmin_data(self, genmin_profile):
        '''
        Get generation capacity data from generation capacity profile.

        Parameters
        ----------
            genmin_profile (str): Path to generation capacity profile csv file.
            
        Returns
        -------
            gen_min (numpy.ndarray): A 2-d array of generation capacity at each bus
        '''
        self.genmin_profile = genmin_profile[self.start_datetime:self.end_datetime].to_numpy()

    def get_genramp_data(self, ramp_profile, interval='30min'):
        '''
        Get ramp rate data from ramp rate profile.

        Parameters
        ----------
            ramp_profile (str): Path to ramp rate profile csv file.
            
        Returns
        -------
            ramp_up (numpy.ndarray): A 2-d array of ramp rate at each bus
            ramp_down (numpy.ndarray): A 2-d array of ramp rate at each bus
        '''
        # Convert 30min ramp rate to hourly ramp rate
        if interval == '30min':
            ramp_profile = ramp_profile*2
        self.genramp_profile = ramp_profile[self.start_datetime:self.end_datetime].to_numpy()

    def get_gencost_data(self, gencost0_profile, gencost1_profile):
        '''
        Get generation cost data from generation cost profile.

        Parameters
        ----------
            gencost0_profile (pandas.DataFrame): A 2-d array of generation cost at each bus
            gencost1_profile (pandas.DataFrame): A 2-d array of generation cost at each bus
        '''
        self.gencost0_profile = gencost0_profile[self.start_datetime:self.end_datetime].to_numpy()
        self.gencost1_profile = gencost1_profile[self.start_datetime:self.end_datetime].to_numpy()

    def get_gen_init_data(self, gen_init):
        '''
        Get generator initial condition.

        Parameters
        ----------
            gen_init (numpy.ndarray): A 1-d array of generator initial condition

        '''

        if gen_init is not None:
            self.gen_init = gen_init/self.baseMVA
        else:
            self.gen_init = None

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
        ##### Constant data
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
        # self.ppc_int = self.ppc_dc

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
        self.gen_i2e = self.ppc_int['order']['gen']['i2e']
        self.dc_idx_f = self.gen_i2e[self.NG-self.num_dcline*2: self.NG-self.num_dcline]
        self.dc_idx_t = self.gen_i2e[self.NG-self.num_dcline: self.NG]
        self.gen_idx_non_dc = self.gen_i2e[:self.NG-self.num_dcline*2]

        # Get mapping from load to bus
        self.load_map = np.zeros((self.NB, self.NL))
        self.load_bus = np.nonzero(self.bus[:, PD])[0]
        for i in range(len(self.load_bus)):
            self.load_map[self.load_bus[i], i] = 1

        # Line flow limit in p.u.
        self.br_max = self.branch[:, RATE_A]/self.baseMVA
        # Replace default value 0 to 999
        self.br_max[self.br_max == 0] = 999.99
        self.br_min = - self.br_max

        # Get interface limit information
        self.if_map = self.ppc_int['if']['map']
        self.if_lims = self.ppc_int['if']['lims']
        self.if_lims[:,1:] = self.if_lims[:,1:]/self.baseMVA
        br_dir, br_idx = np.sign(self.if_map[:,1]), np.abs(self.if_map[:,1]).astype(int)
        self.if_map[:, 1] = br_dir*(br_idx-1)

        ##### User defined data
        # Historical generation data
        if self.gen_profile is not None:
            self.gen_hist = np.empty((self.NT, self.NG))
            self.gen_hist[:, self.gen_idx_non_dc] = self.gen_profile/self.baseMVA
            self.gen_hist[:, self.dc_idx_f] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_f, PG]/self.baseMVA
            self.gen_hist[:, self.dc_idx_t] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_t, PG]/self.baseMVA
        else:
            self.gen_hist = np.zeros((self.NT, self.NG))

        # Generator upper operating limit in p.u.
        if self.genmax_profile is not None:
            self.gen_max = np.empty((self.NT, self.NG))
            self.gen_max[:, self.gen_idx_non_dc] = self.genmax_profile/self.baseMVA
            self.gen_max[:, self.dc_idx_f] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_f, PMAX]/self.baseMVA
            self.gen_max[:, self.dc_idx_t] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_t, PMAX]/self.baseMVA
        else:
            self.gen_max = np.ones((self.NT, self.NG))*self.gen[:, PMAX]/self.baseMVA

         # Generator lower operating limit in p.u.
        if self.genmin_profile is not None:
            self.gen_min = np.empty((self.NT, self.NG))
            self.gen_min[:, self.gen_idx_non_dc] = self.genmin_profile/self.baseMVA
            self.gen_min[:, self.dc_idx_f] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_f, PMIN]/self.baseMVA
            self.gen_min[:, self.dc_idx_t] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_t, PMIN]/self.baseMVA
        else:
            self.gen_min = np.ones((self.NT, self.NG))*self.gen[:, PMIN]/self.baseMVA

        # Generator ramp rate limit in p.u.
        if self.genramp_profile is not None:         
            self.ramp_up = np.empty((self.NT, self.NG))
            self.ramp_up[:, self.gen_idx_non_dc] = self.genramp_profile/self.baseMVA
            self.ramp_up[:, self.dc_idx_f] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_f, RAMP_30]*2/self.baseMVA
            self.ramp_up[:, self.dc_idx_t] = np.ones((self.NT, self.num_dcline))*self.gen[self.dc_idx_t, RAMP_30]*2/self.baseMVA

            self.ramp_down = np.min([self.gen_max, self.ramp_up], axis=0)
        else:
            self.ramp_up = np.ones((self.NT, self.NG))*self.gen[:, RAMP_30]*2/self.baseMVA
            self.ramp_down = np.min([self.gen_max, self.ramp_up], axis=0)

        # Linear cost intercept coefficients in p.u.
        if self.gencost0_profile is not None:           
            self.gencost_0 = np.empty((self.NT, self.NG))
            self.gencost_0[:, self.gen_idx_non_dc] = self.gencost0_profile
            self.gencost_0[:, self.dc_idx_f] = np.ones((self.NT, self.num_dcline))*self.gencost[self.dc_idx_f, COST+1]
            self.gencost_0[:, self.dc_idx_t] = np.ones((self.NT, self.num_dcline))*self.gencost[self.dc_idx_t, COST+1]

        else:
            self.gencost_0 = np.ones((self.NT, self.NG))*self.gencost[:, COST+1]

        # Linear cost slope coefficients in p.u.
        if self.gencost1_profile is not None:           
            self.gencost_1 = np.empty((self.NT, self.NG))
            self.gencost_1[:, self.gen_idx_non_dc] = self.gencost1_profile*self.baseMVA
            self.gencost_1[:, self.dc_idx_f] = np.ones((self.NT, self.num_dcline))*self.gencost[self.dc_idx_f, COST]*self.baseMVA
            self.gencost_1[:, self.dc_idx_t] = np.ones((self.NT, self.num_dcline))*self.gencost[self.dc_idx_t, COST]*self.baseMVA
        else:
            self.gencost_1 = np.ones((self.NT, self.NG))*self.gencost[:, COST]*self.baseMVA
            
        # Convert load to p.u.
        if self.load_profile is not None:
            self.load_pu = self.load_profile/self.baseMVA
        else:
            self.load_pu = np.ones((self.NT, self.NL))*self.bus[:, PD]/self.baseMVA
            Warning("No load profile is provided. Using default load profile.")

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
        dcline_gen = np.zeros((num_dcline*2, 21))
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
        dcline_gen[:, PMAX] = np.concatenate([dcline[:, DC_PMAX],
                                            dcline[:, DC_PMAX]])
        dcline_gen[:, PMIN] = np.concatenate([dcline[:, DC_PMIN],
                                            dcline[:, DC_PMIN]])
        dcline_gen[:, RAMP_AGC] = np.ones(num_dcline*2)*1e10 # Unlimited ramp rate
        dcline_gen[:, RAMP_10] = np.ones(num_dcline*2)*1e10 # Unlimited ramp rate
        dcline_gen[:, RAMP_30] = np.ones(num_dcline*2)*1e10 # Unlimited ramp rate
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
        results_lmp = bus_lmp.reshape(self.NT, self.NB)
        results_lmp = pd.DataFrame(results_lmp, index=self.timestamp_list)

        results = {
            'PG': results_pg,
            'VA': results_va,
            'PF': results_pf,
            'IF': results_if,
            'LMP': results_lmp
        }

        return results

    def show_model_dim(self, model_multi_opf):
        '''
        Show model dimensions.
        '''
        print('Number of buses: {}'.format(self.NB))
        print('Number of generators: {}'.format(self.NG))
        print('Number of branches: {}'.format(self.NBR))
        print('Number of time periods: {}'.format(self.NT))

        num_vars = len(model_multi_opf.PG) \
                    + len(model_multi_opf.Va) \
                    + len(model_multi_opf.PF)
        print('Number of variables: {}'.format(num_vars))

        num_constraints = len(model_multi_opf.c_br_flow) \
                    + len(model_multi_opf.c_br_max) \
                    + len(model_multi_opf.c_br_min) \
                    + len(model_multi_opf.c_gen_max) \
                    + len(model_multi_opf.c_gen_min) \
                    + len(model_multi_opf.c_pf) \
                    + len(model_multi_opf.c_gen_ramp_up) \
                    + len(model_multi_opf.c_gen_ramp_down) \
                    + len(model_multi_opf.c_dcline) \
                    + len(model_multi_opf.c_if_max) \
                    + len(model_multi_opf.c_if_min)
        print('Number of constraints: {}'.format(num_constraints))

    def get_last_gen(self, model_multi_opf):
        '''
        Get generator power generation at the last simulation.
        Used to create initial condition for the next simulation.
        '''
        # Get dimensions of the last simulation
        NT = len(model_multi_opf.PG_index_0)
        NG = len(model_multi_opf.PG_index_1)
        results_pg = np.array(model_multi_opf.PG[:,:]()).reshape(NT, NG)*self.baseMVA
        last_gen = results_pg[-1, :]
        return last_gen
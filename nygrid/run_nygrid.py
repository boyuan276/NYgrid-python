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

    def run_single_opf(self):
        pass

    def run_multi_opf(self):
        pass

    def get_load_data(self):
        pass

    def process_ppc(self, load):
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
        self.ppc_dc, self.num_dcline = self.dcline2gen(self.ppc)

        # Convert to internal indexing
        self.ppc_int = pp.ext2int(self.ppc_dc)

        self.baseMVA, self.bus, self.gen, self.branch, self.gencost = \
            (self.ppc_int["baseMVA"], self.ppc_int["bus"], 
             self.ppc_int["gen"], self.ppc_int["branch"], 
             self.ppc_int["gencost"])

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
        dc_idx_f = gen_i2e[self.NG-self.num_dcline*2: self.NG-self.num_dcline]
        dc_idx_t = gen_i2e[self.NG-self.num_dcline: self.NG]

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
        self.load = load/self.baseMVA

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
    
    def dcline2gen(ppc):
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
        baseMVA, gen, gencost, dcline, genfuel = \
            ppc["baseMVA"], ppc["gen"], ppc["gencost"], ppc['dcline'], ppc['genfuel']

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
        # dcline_gen[:, PMAX] = np.concatenate([dcline[:, DC_PMAX],
        #                                     dcline[:, DC_PMAX]])
        dcline_gen[:, PMAX] = np.full(num_dcline*2, np.inf)
        dcline_gen[:, PMIN] = np.concatenate([dcline[:, DC_PMIN],
                                            dcline[:, DC_PMIN]])
        # Add the DC line converted generators to the gen matrix
        ppc['gen'] = np.concatenate([gen, dcline_gen])

        # Set gencost parameters of the DC line converted generators
        dcline_gencost = np.zeros((num_dcline*2, 6))
        dcline_gencost[:, MODEL] = np.ones(num_dcline*2)*POLYNOMIAL
        dcline_gencost[:, NCOST] = np.ones(num_dcline*2)*2
        # Add the DC line converted generators to the gencost matrix
        ppc['gencost'] = np.concatenate([gencost, dcline_gencost])

        # Add the DC line converted generators to the genfuel list
        dcline_genfuel = ['dc line']*num_dcline*2
        ppc['genfuel'] = np.concatenate([genfuel, dcline_genfuel])

        return ppc, num_dcline

"""
Class for running the NYGrid model.

Known Issues/Wishlist:
1. Better dc line model
2. Better documentation
3. Check dim of start/end datetime and load profile

"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import pandas as pd
import pypower.api as pp
from nygrid.ppc_idx import *
from nygrid.utlis import format_date
from nygrid.optimizer import Optimizer
import logging


def check_status(results):
    """
    Check the status of a Pyomo model.

    Parameters
    ----------
    results: pyomo.opt.results.results_.SolverResults
        Pyomo model results.

    Returns
    -------
    status: bool
        True if the model is solved successfully.
    """

    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        status = True
        print("The problem is feasible and optimal!")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        status = False
        raise RuntimeError("The problem is infeasible!")
    else:
        status = False
        print(str(results.solver))
        raise RuntimeError("Something else is wrong!")
    return status


def convert_dcline_2_gen(ppc, dcline_prop=None):
    """
    Convert DC lines to generators and add their parameters in the PyPower matrices.

    Parameters
    ----------
    ppc: dict
        PyPower case dictionary.
    dcline_prop: numpy.ndarray
        DC line properties.

    Returns
    -------
    ppc_dc: dict
        PyPower case dictionary with DC lines converted to generators.
    num_dcline: int
        Number of DC lines.
        ppc (dict): PyPower case dictionary.
    """

    # TODO: Use dcline prop from an external file

    # Get PyPower case information
    ppc_dc = ppc.copy()
    baseMVA = ppc_dc['baseMVA']
    gen = ppc_dc['gen']
    gencost = ppc_dc['gencost']
    genfuel = ppc_dc['genfuel']

    if dcline_prop is not None and dcline_prop.size > 0:
        dcline = dcline_prop
    else:
        dcline = ppc_dc['dcline']
        Warning('No DC line properties are provided. Use default values from the PyPower case.')

    # Set gen parameters of the DC line converted generators
    num_dcline = dcline.shape[0]
    dcline_gen = np.zeros((num_dcline * 2, 21))  # One for from bus, one for to bus
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
    dcline_gen[:, MBASE] = np.ones(num_dcline * 2) * baseMVA
    dcline_gen[:, GEN_STATUS] = np.concatenate([dcline[:, DC_BR_STATUS],
                                                dcline[:, DC_BR_STATUS]])
    dcline_gen[:, PMAX] = np.concatenate([dcline[:, DC_PMAX],
                                          dcline[:, DC_PMAX]])
    dcline_gen[:, PMIN] = np.concatenate([dcline[:, DC_PMIN],
                                          dcline[:, DC_PMIN]])
    dcline_gen[:, RAMP_AGC] = np.ones(num_dcline * 2) * 1e10  # Unlimited ramp rate
    dcline_gen[:, RAMP_10] = np.ones(num_dcline * 2) * 1e10  # Unlimited ramp rate
    dcline_gen[:, RAMP_30] = np.ones(num_dcline * 2) * 1e10  # Unlimited ramp rate
    # Add the DC line converted generators to the gen matrix
    ppc_dc['gen'] = np.concatenate([gen, dcline_gen])

    # Set gencost parameters of the DC line converted generators
    dcline_gencost = np.zeros((num_dcline * 2, 6))
    dcline_gencost[:, MODEL] = np.ones(num_dcline * 2) * POLYNOMIAL
    dcline_gencost[:, NCOST] = np.ones(num_dcline * 2) * 2
    # Add the DC line converted generators to the gencost matrix
    ppc_dc['gencost'] = np.concatenate([gencost, dcline_gencost])

    # Add the DC line converted generators to the genfuel list
    dcline_genfuel = np.array(['DC Line From'] * num_dcline
                           + ['DC Line To'] * num_dcline)
    ppc_dc['genfuel'] = np.concatenate([genfuel, dcline_genfuel])

    return ppc_dc, num_dcline


def convert_esr_2_gen(ppc, esr_prop=None):
    """
    Convert ESR to generators and add their parameters in the PyPower matrices.

    Parameters
    ----------
    ppc: dict
        PyPower case dictionary.
    esr_prop: numpy.ndarray
        ESR properties.

    Returns
    -------
    ppc_esr: dict
        PyPower case dictionary with ESR converted to generators.
    num_esr: int
        Number of ESR.
    """

    # Get PyPower case information
    ppc_esr = ppc.copy()
    baseMVA = ppc_esr['baseMVA']
    gen = ppc_esr['gen']
    gencost = ppc_esr['gencost']
    genfuel = ppc_esr['genfuel']

    if esr_prop is None or esr_prop.size == 0:
        Warning('No ESR properties are provided.')
        return ppc_esr, 0

    # Set gen parameters of the ESR converted generators
    num_esr = esr_prop.shape[0]
    esr_gen = np.zeros((num_esr * 2, 21))  # One for charging, one for discharging
    esr_gen[:, GEN_BUS] = np.concatenate([esr_prop[:, ESR_BUS],
                                          esr_prop[:, ESR_BUS]])
    esr_gen[:, PG] = np.concatenate([esr_prop[:, ESR_CRG_MAX],
                                     -esr_prop[:, ESR_DIS_MAX]])
    esr_gen[:, QG] = np.zeros(num_esr * 2)
    esr_gen[:, QMAX] = np.zeros(num_esr * 2)
    esr_gen[:, QMIN] = np.zeros(num_esr * 2)
    esr_gen[:, VG] = np.ones(num_esr * 2)
    esr_gen[:, MBASE] = np.ones(num_esr * 2) * baseMVA
    esr_gen[:, GEN_STATUS] = np.ones(num_esr * 2)
    esr_gen[:, PMAX] = np.concatenate([esr_prop[:, ESR_CRG_MAX],
                                       esr_prop[:, ESR_DIS_MAX]])
    esr_gen[:, PMIN] = np.zeros(num_esr * 2)
    esr_gen[:, RAMP_AGC] = np.ones(num_esr * 2) * 1e10  # Unlimited ramp rate
    esr_gen[:, RAMP_10] = np.ones(num_esr * 2) * 1e10  # Unlimited ramp rate
    esr_gen[:, RAMP_30] = np.ones(num_esr * 2) * 1e10  # Unlimited ramp rate
    # Add the ESR converted generators to the gen matrix
    ppc_esr['gen'] = np.concatenate([gen, esr_gen])

    # Set gencost parameters of the ESR converted generators
    esr_gencost = np.zeros((num_esr * 2, 6))
    esr_gencost[:, MODEL] = np.ones(num_esr * 2) * POLYNOMIAL
    esr_gencost[:, NCOST] = np.ones(num_esr * 2) * 2
    # Add the ESR converted generators to the gencost matrix
    ppc_esr['gencost'] = np.concatenate([gencost, esr_gencost])

    # Add the ESR converted generators to the genfuel list
    esr_genfuel = np.array(['ESR Charging'] * num_esr
                           + ['ESR Discharging'] * num_esr)
    ppc_esr['genfuel'] = np.concatenate([genfuel, esr_genfuel])

    return ppc_esr, num_esr


class NYGrid:
    """
    Class for running the NYGrid model.

    """

    def __init__(self, ppc_filename, start_datetime, end_datetime,
                 dcline_prop=None, esr_prop=None, verbose=False):
        """
        Initialize the NYGrid model.

        Parameters
        ----------
        ppc_filename: str
            Path to the PyPower case file.
        start_datetime: str
            Start datetime of the simulation.
        end_datetime: str
            End datetime of the simulation.
        verbose: bool
            If True, print out the information of the simulation.

        Returns
        -------
        None
        """

        # %% Load PyPower case
        self.ppc = pp.loadcase(ppc_filename)

        # %% Set the start and end datetime of the simulation
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.verbose = verbose

        # Format the forecast start/end and determine the total time.
        self.start_datetime = format_date(start_datetime)
        self.end_datetime = format_date(end_datetime)
        self.delta_t = self.end_datetime - self.start_datetime
        self.timestamp_list = pd.date_range(self.start_datetime, self.end_datetime, freq='1H')
        self.NT = len(self.timestamp_list)

        if self.verbose:
            logging.info('Initializing NYGrid run...')
            logging.info(f'NYGrid run starting on: {self.start_datetime}')
            logging.info(f'NYGrid run ending on: {self.end_datetime}')
            logging.info(f'NYGrid run duration: {self.delta_t}')

        # %% Process PyPower case constant fields
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
        self.ppc_dc, self.NDCL = convert_dcline_2_gen(self.ppc, dcline_prop)

        # Convert ESR to generators and add to gen matrix
        self.ppc_dc_esr, self.NESR = convert_esr_2_gen(self.ppc_dc, esr_prop)

        # TODO: Change indexing from ppc_dc to ppc_dc_esr

        # Convert to internal indexing
        self.ppc_int = pp.ext2int(self.ppc_dc_esr)
        # self.ppc_int = self.ppc_dc

        self.baseMVA = self.ppc_int['baseMVA']
        self.bus = self.ppc_int['bus']
        self.gen = self.ppc_int['gen']
        self.branch = self.ppc_int['branch']
        self.gencost = self.ppc_int['gencost']

        # Generator info
        self.gen_bus = self.gen[:, GEN_BUS].astype(int)  # what buses are they at?

        # Build B matrices and phase shift injections
        B, Bf, _, _ = pp.makeBdc(self.baseMVA, self.bus, self.branch)
        self.B = B.todense()
        self.Bf = Bf.todense()

        # Linear shift factor
        self.PTDF = pp.makePTDF(self.baseMVA, self.bus, self.branch)

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
        self.dc_idx_f = self.gen_i2e[self.NG - self.NDCL * 2: self.NG - self.NDCL]
        self.dc_idx_t = self.gen_i2e[self.NG - self.NDCL: self.NG]
        self.gen_idx_non_dc = self.gen_i2e[:self.NG - self.NDCL * 2]

        # Get mapping from load to bus
        self.load_map = np.zeros((self.NB, self.NL))
        self.load_bus = np.nonzero(self.bus[:, PD])[0]
        for i in range(len(self.load_bus)):
            self.load_map[self.load_bus[i], i] = 1

        # Line flow limit in p.u.
        self.br_max = self.branch[:, RATE_A] / self.baseMVA
        # Replace default value 0 to 999.99
        self.br_max[self.br_max == 0] = 999.99
        self.br_min = - self.br_max

        # Get interface limit information
        self.if_map = self.ppc_int['if']['map']
        self.if_lims = self.ppc_int['if']['lims']
        self.if_lims[:, 1:] = self.if_lims[:, 1:] / self.baseMVA
        br_dir, br_idx = np.sign(self.if_map[:, 1]), np.abs(self.if_map[:, 1]).astype(int)
        self.if_map[:, 1] = br_dir * (br_idx - 1)
        self.NIF = len(self.if_lims)

        self.if_br_dir = np.empty(self.NIF, dtype=object)
        self.if_br_idx = np.empty(self.NIF, dtype=object)
        self.if_lims_max = np.empty(self.NIF, dtype=float)
        self.if_lims_min = np.empty(self.NIF, dtype=float)

        for n in range(self.NIF):
            if_id, if_lims_min, if_lims_max = self.if_lims[n, :]
            br_dir_idx = self.if_map[(self.if_map[:, 0] == int(if_id)), 1]
            br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
            self.if_br_dir[n] = br_dir
            self.if_br_idx[n] = br_idx
            self.if_lims_max[n] = if_lims_max
            self.if_lims_min[n] = if_lims_min

        # Historical generation data. Use zero as default values
        self.gen_hist = np.zeros((self.NT, self.NG))

        # Generator upper operating limit in p.u.
        self.gen_max = np.ones((self.NT, self.NG)) * self.gen[:, PMAX] / self.baseMVA

        # Generator lower operating limit in p.u.
        self.gen_min = np.ones((self.NT, self.NG)) * self.gen[:, PMIN] / self.baseMVA

        # Generator ramp rate limit in p.u./hour
        self.ramp_up = np.ones((self.NT, self.NG)) * self.gen[:, RAMP_30] * 2 / self.baseMVA
        # Downward ramp rate is the minimum of the upward ramp rate and the maximum generation limit
        self.ramp_down = np.min([self.gen_max, self.ramp_up], axis=0)

        # Linear cost intercept coefficients in p.u.
        self.gencost_0 = np.ones((self.NT, self.NG)) * self.gencost[:, COST + 1]

        # Linear cost slope coefficients in p.u.
        self.gencost_1 = np.ones((self.NT, self.NG)) * self.gencost[:, COST] * self.baseMVA

        # Convert load to p.u.
        self.load_pu = np.ones((self.NT, self.NL)) * self.bus[:, PD] / self.baseMVA

        # Generator initial condition
        self.gen_init = None

        # %% Create Pyomo model
        self.model = None

        # %% Penalty parameters and default values
        # ED module
        self.NoPowerBalanceViolation = False
        self.NoRampViolation = False
        self.PenaltyForOverGeneration = 1_500  # $/MWh
        self.PenaltyForLoadShed = 5_000  # $/MWh
        self.PenaltyForRampViolation = 11_000  # $/MW
        # UC module
        self.PenaltyForMinTimeViolation = 1_000  # $/MWh, Not used
        self.PenaltyForNumberCommitViolation = 10_000  # $/hour, Not used
        # RS module
        self.NoReserveViolation = False
        self.PenaltyForReserveViolation = 1_300  # $/MW, Not used
        self.NoImplicitReserveCascading = False
        self.OfflineReserveNotFromOnline = False
        # PF module
        self.NoPowerflowViolation = False
        self.HvdcHurdleCost = 0.10  # $/MWh, Not used
        self.PenaltyForBranchMwViolation = 1_000  # $/MWh
        self.PenaltyForInterfaceMWViolation = 1_000  # $/MWh
        self.MaxPhaseAngleDifference = 1.5  # Radians, Not used
        # ES module
        self.PenaltyForEnergyStorageViolation = 8_000  # $/MWh, Not used

        # Model options
        # Use Power Transfer Distribution Factors (PTDF) for linear shift factor
        self.UsePTDF = True
        self.solver = 'gurobi'

    def set_options(self, options):

        # TODO: Check if this is the right way to set the penalty parameters
        for key, value in options.items():
            setattr(self, key, value)
            logging.info(f'Set {key} to {value} ...')

    def set_load_sch(self, load_sch):
        """
        Set load schedule data from load profile.

        Parameters
        ----------
        load_sch: pandas.DataFrame
            Load profile of the network.

        Returns
        -------
        None
        """

        # Slice the load profile to the simulation period
        load_sch = load_sch[self.start_datetime:self.end_datetime].to_numpy()

        # Convert load to p.u.
        if load_sch is not None and load_sch.size > 0:
            self.load_pu = load_sch / self.baseMVA
        else:
            raise ValueError('No load profile is provided.')

    def set_gen_mw_sch(self, gen_mw_sch):
        """
        Set generator schedule data from generation profile.

        Parameters
        ----------
        gen_mw_sch: pandas.DataFrame
            Generation profile of thermal generators.

        Returns
        -------
        None
        """

        # Slice the generation profile to the simulation period
        gen_mw_sch = gen_mw_sch[self.start_datetime:self.end_datetime].to_numpy()

        # Generator schedule in p.u.
        if gen_mw_sch is not None and gen_mw_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.gen_hist = np.empty((self.NT, self.NG))
            self.gen_hist[:, self.gen_idx_non_dc] = gen_mw_sch / self.baseMVA
            # HVDC Proxy generators: Use default values from the PyPower case
            self.gen_hist[:, self.dc_idx_f] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_f, PG] / self.baseMVA
            self.gen_hist[:, self.dc_idx_t] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_t, PG] / self.baseMVA
        else:
            raise ValueError('No generation profile is provided.')

    def set_gen_max_sch(self, gen_max_sch):
        """
        Set generator upper operating limit data from generation capacity profile.

        Parameters
        ----------
        gen_max_sch: pandas.DataFrame
            Generator upper operating limit profile of thermal generators.

        Returns
        -------
        None
        """

        # Slice the generator profile to the simulation period
        gen_max_sch = gen_max_sch[self.start_datetime:self.end_datetime].to_numpy()

        # Generator upper operating limit in p.u.
        if gen_max_sch is not None and gen_max_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.gen_max = np.empty((self.NT, self.NG))
            self.gen_max[:, self.gen_idx_non_dc] = gen_max_sch / self.baseMVA
            # HVDC Proxy generators: Use default values from the PyPower case
            self.gen_max[:, self.dc_idx_f] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_f, PMAX] / self.baseMVA
            self.gen_max[:, self.dc_idx_t] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_t, PMAX] / self.baseMVA
        else:
            raise ValueError('No generation capacity profile is provided.')

    def set_gen_min_sch(self, gen_min_sch):
        """
        Set generator lower operating limit data from generation capacity profile.

        Parameters
        ----------
        gen_min_sch: pandas.DataFrame
            Generator lower operating limit profile of thermal generators.

        Returns
        -------
        None
        """

        # Slice the generator profile to the simulation period
        gen_min_sch = gen_min_sch[self.start_datetime:self.end_datetime].to_numpy()

        # Generator lower operating limit in p.u.
        if gen_min_sch is not None and gen_min_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.gen_min = np.empty((self.NT, self.NG))
            self.gen_min[:, self.gen_idx_non_dc] = gen_min_sch / self.baseMVA
            # HVDC Proxy generators: Use default values from the PyPower case
            self.gen_min[:, self.dc_idx_f] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_f, PMIN] / self.baseMVA
            self.gen_min[:, self.dc_idx_t] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_t, PMIN] / self.baseMVA
        else:
            raise ValueError('No generation capacity profile is provided.')

    def set_gen_ramp_sch(self, gen_ramp_sch, interval='30min'):
        """
        Set generator ramp rate limit data from ramp rate profile.

        Parameters
        ----------
        gen_ramp_sch: pandas.DataFrame
            Generator ramp rate limit profile of thermal generators.
        interval: str
            Time interval of the ramp rate profile. Default is 30min.

        Returns
        -------
        None
        """

        # Convert 30min ramp rate to hourly ramp rate
        if interval == '30min':
            gen_ramp_sch = gen_ramp_sch * 2
            gen_ramp_sch = gen_ramp_sch[self.start_datetime:self.end_datetime].to_numpy()
        else:
            raise ValueError('Only support 30min ramp rate profile.')

        # Generator ramp rate limit in p.u./hour
        if gen_ramp_sch is not None and gen_ramp_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.ramp_up = np.empty((self.NT, self.NG))
            self.ramp_up[:, self.gen_idx_non_dc] = gen_ramp_sch / self.baseMVA
            # HVDC Proxy generators: Use default values from the PyPower case
            self.ramp_up[:, self.dc_idx_f] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_f, RAMP_30] * 2 / self.baseMVA
            self.ramp_up[:, self.dc_idx_t] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dc_idx_t, RAMP_30] * 2 / self.baseMVA
        else:
            raise ValueError('No ramp rate profile is provided.')

        # Downward ramp rate is the minimum of the upward ramp rate and the maximum generation limit
        self.ramp_down = np.min([self.gen_max, self.ramp_up], axis=0)

    def set_gen_cost_sch(self, gen_cost0_sch, gen_cost1_sch):
        """
        Set generator cost data from generation cost profile.

        Parameters
        ----------
        gen_cost0_sch: pandas.DataFrame
            Generator cost intercept profile of thermal generators.
        gen_cost1_sch: pandas.DataFrame
            Generator cost slope profile of thermal generators.

        Returns
        -------
        None
        """

        # Slice the generator profile to the simulation period
        gen_cost0_sch = gen_cost0_sch[self.start_datetime:self.end_datetime].to_numpy()
        gen_cost1_sch = gen_cost1_sch[self.start_datetime:self.end_datetime].to_numpy()

        # Linear cost intercept coefficients in p.u.
        if gen_cost0_sch is not None and gen_cost0_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.gencost_0 = np.empty((self.NT, self.NG))
            self.gencost_0[:, self.gen_idx_non_dc] = gen_cost0_sch
            # HVDC Proxy generators: Use default values from the PyPower case
            self.gencost_0[:, self.dc_idx_f] = np.ones(
                (self.NT, self.NDCL)) * self.gencost[self.dc_idx_f, COST + 1]
            self.gencost_0[:, self.dc_idx_t] = np.ones(
                (self.NT, self.NDCL)) * self.gencost[self.dc_idx_t, COST + 1]
        else:
            raise ValueError('No generation cost profile is provided.')

        # Linear cost slope coefficients in p.u.
        if gen_cost1_sch is not None and gen_cost1_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.gencost_1 = np.empty((self.NT, self.NG))
            self.gencost_1[:, self.gen_idx_non_dc] = gen_cost1_sch * self.baseMVA
            # HVDC Proxy generators: Use default values from the PyPower case
            self.gencost_1[:, self.dc_idx_f] = np.ones(
                (self.NT, self.NDCL)) * self.gencost[self.dc_idx_f, COST] * self.baseMVA
            self.gencost_1[:, self.dc_idx_t] = np.ones(
                (self.NT, self.NDCL)) * self.gencost[self.dc_idx_t, COST] * self.baseMVA
        else:
            raise ValueError('No generation cost profile is provided.')

    def set_gen_init_data(self, gen_init):
        """
        Get generator initial condition.

        Parameters
        ----------
            gen_init (numpy.ndarray): A 1-d array of generator initial condition

        """

        if gen_init is not None and gen_init.size > 0:
            self.gen_init = gen_init / self.baseMVA
        else:
            Warning('No generator initial condition is provided.')

    def check_input_dim(self):
        """
        Check the dimensions of the input data.

        Returns
        -------
        None
        """
        if (self.gen_min.shape != self.gen_max.shape) \
                or (self.ramp_down.shape != self.ramp_up.shape):
            raise ValueError('Found mismatch in generator constraint dimensions!')

        if self.br_min.shape != self.br_max.shape:
            raise ValueError('Found mismatch in branch flow limit array dimensions!')

    def create_dc_opf(self):
        """
        Create a multi-period DC OPF problem.

        Returns
        -------
        None
        """

        # Create optimizer
        optimizer = Optimizer(self)

        # Add variables
        optimizer.add_vars_ed()
        optimizer.add_vars_pf()
        optimizer.add_vars_dual()

        # Add constraints
        optimizer.add_constrs_ed()
        optimizer.add_constrs_pf()

        # Add objective
        optimizer.add_obj()

        self.model = optimizer.model

    def solve_dc_opf(self, solver_options=None):

        # Create Pyomo model
        self.model = pyo.ConcreteModel(name='multi-period DC OPF')

        # Check input dimensions
        self.check_input_dim()

        # Create optimizer
        self.create_dc_opf()

        # Show model dimensions
        if self.verbose:
            self.show_model_dim()

        # Solve the optimization problem
        opt = pyo.SolverFactory(self.solver)
        if solver_options is not None:
            opt.options.update(solver_options)
        results = opt.solve(self.model, tee=self.verbose)

        # Check the status of the optimization problem
        if check_status(results):
            print(f"Objective function value: {self.model.obj():.3e}")

    def get_results_dc_opf(self):
        """
        Get results for a multi-period OPF problem.

        Returns:
            results (dict): a dict of pandas DataFrames, including:
                1. Generator power generation.
                2. Bus phase angle.
                3. Branch power flow.
                4. Interface flow.
                5. Bus locational marginal price (LMP).
                6. Total cost.
        """

        # Create dictionary to store results
        vars = dict()

        # Power generation
        results_pg = np.array(self.model.PG[:, :]()).reshape(self.NT, self.NG) * self.baseMVA
        gen_order = self.ppc_int['order']['gen']['e2i']
        results_pg = pd.DataFrame(results_pg, index=self.timestamp_list,
                                  columns=gen_order).sort_index(axis=1)
        vars['PG'] = results_pg

        # Branch power flow
        branch_pf = np.array(self.model.PF[:, :]()).reshape(self.NT, self.NBR) * self.baseMVA
        results_pf = pd.DataFrame(branch_pf, index=self.timestamp_list)
        vars['PF'] = results_pf

        # Interface flow
        if_flow = np.zeros((self.NT, self.NIF))
        for t in range(self.NT):
            for n in range(self.NIF):
                br_dir = self.if_br_dir[n]
                br_idx = self.if_br_idx[n]
                if_flow[t, n] = sum(br_dir[i] * branch_pf[t, br_idx[i] - 1]
                                    for i in range(len(br_idx)))
                if_flow[t, n] = sum(br_dir[i] * branch_pf[t, br_idx[i] - 1]
                                    for i in range(len(br_idx)))
        results_if = pd.DataFrame(if_flow, index=self.timestamp_list)
        vars['IF'] = results_if

        if not self.UsePTDF:
            # Bus phase angle
            results_va = np.array(self.model.VA[:, :]()).reshape(self.NT, self.NB) * 180 / np.pi
            # Just to compare with PyPower
            results_va = results_va - 73.4282
            # Convert negative numbers to 0-360
            results_va = np.where(results_va < 0, results_va + 360, results_va)
            results_va = pd.DataFrame(results_va, index=self.timestamp_list)
            vars['VA'] = results_va

            # Bus locational marginal price (LMP)
            results_lmp = np.zeros((self.NT, self.NB))
            for t in range(self.NT):
                for n in range(self.NB):
                    results_lmp[t, n] = np.abs(self.model.dual[self.model.c_pf[t, n]]) / self.baseMVA
            results_lmp = pd.DataFrame(results_lmp, index=self.timestamp_list)
            vars['LMP'] = results_lmp

        else:
            # Bus power injection
            results_pbus = np.array(self.model.PBUS[:, :]()).reshape(self.NT, self.NB) * self.baseMVA
            results_pbus = pd.DataFrame(results_pbus, index=self.timestamp_list)
            vars['PBUS'] = results_pbus

            # Bus locational marginal price (LMP)
            results_lmp = np.zeros((self.NT, self.NB))
            for t in range(self.NT):
                for n in range(self.NL):
                    results_lmp[t, n] = np.abs(self.model.dual[self.model.c_load_set[t, n]]) / self.baseMVA
            # sp_energy = np.zeros(self.NT)  # Shadow price of energy
            # sp_congestion_max = np.zeros((self.NT, self.NB))  # Shadow price of congestion max
            # sp_congestion_min = np.zeros((self.NT, self.NB))  # Shadow price of congestion min
            # for t in range(self.NT):
            #     sp_energy[t] = np.abs(self.model.dual[self.model.c_energy_balance[t]]) / self.baseMVA
            #     for n in range(self.NB):
            #         sp_congestion_max[t, n] = np.abs(self.model.dual[self.model.c_br_max[t, n]]) / self.baseMVA
            #         sp_congestion_min[t, n] = np.abs(self.model.dual[self.model.c_br_min[t, n]]) / self.baseMVA
            #         results_lmp[t, n] = sp_energy[t] + sp_congestion_max[t, n] + sp_congestion_min[t, n]
            results_lmp = pd.DataFrame(results_lmp, index=self.timestamp_list)
            vars['LMP'] = results_lmp

        # Slack variables
        results_s_ramp_down = np.array(self.model.s_ramp_down[:, :]()).reshape(self.NT, self.NG) * self.baseMVA
        results_s_ramp_up = np.array(self.model.s_ramp_up[:, :]()).reshape(self.NT, self.NG) * self.baseMVA
        results_s_over_gen = np.array(self.model.s_over_gen[:]()) * self.baseMVA
        results_s_load_shed = np.array(self.model.s_load_shed[:]()) * self.baseMVA
        results_s_if_max = np.array(self.model.s_if_max[:, :]()).reshape(self.NT, self.NIF) * self.baseMVA
        results_s_if_min = np.array(self.model.s_if_min[:, :]()).reshape(self.NT, self.NIF) * self.baseMVA
        results_s_br_max = np.array(self.model.s_br_max[:, :]()).reshape(self.NT, self.NBR) * self.baseMVA
        results_s_br_min = np.array(self.model.s_br_min[:, :]()).reshape(self.NT, self.NBR) * self.baseMVA

        slack_vars = {
            's_ramp_up': results_s_ramp_up,
            's_ramp_down': results_s_ramp_down,
            's_over_gen': results_s_over_gen,
            's_load_shed': results_s_load_shed,
            's_if_max': results_s_if_max,
            's_if_min': results_s_if_min,
            's_br_max': results_s_br_max,
            's_br_min': results_s_br_min
        }

        # Total cost
        pg_pu = np.array(self.model.PG[:, :]()).reshape(self.NT, self.NG)
        gen_cost = sum(self.gencost_0[t, g] + self.gencost_1[t, g] * pg_pu[t, g]
                       for g in range(self.NG) for t in range(self.NT))
        over_gen_penalty = sum(self.PenaltyForOverGeneration * results_s_over_gen[t]
                               for t in range(self.NT))
        load_shed_penalty = sum(self.PenaltyForLoadShed * results_s_load_shed[t]
                                for t in range(self.NT))
        ramp_up_penalty = sum(self.PenaltyForRampViolation * results_s_ramp_up[t, g]
                              for g in range(self.NG) for t in range(self.NT))
        ramp_down_penalty = sum(self.PenaltyForRampViolation * results_s_ramp_down[t, g]
                                for g in range(self.NG) for t in range(self.NT))
        if_max_penalty = sum(self.PenaltyForInterfaceMWViolation * results_s_if_max[t, n]
                             for n in range(self.NIF) for t in range(self.NT))
        if_min_penalty = sum(self.PenaltyForInterfaceMWViolation * results_s_if_min[t, n]
                             for n in range(self.NIF) for t in range(self.NT))
        br_max_penalty = sum(self.PenaltyForBranchMwViolation * results_s_br_max[t, n]
                             for n in range(self.NBR) for t in range(self.NT))
        br_min_penalty = sum(self.PenaltyForBranchMwViolation * results_s_br_min[t, n]
                             for n in range(self.NBR) for t in range(self.NT))

        total_cost = gen_cost + over_gen_penalty + load_shed_penalty + \
                     ramp_up_penalty + ramp_down_penalty + \
                     if_max_penalty + if_min_penalty + \
                     br_max_penalty + br_min_penalty

        costs = {
            'total_cost': total_cost,
            'gen_cost': gen_cost,
            'over_gen_penalty': over_gen_penalty,
            'load_shed_penalty': load_shed_penalty,
            'ramp_up_penalty': ramp_up_penalty,
            'ramp_down_penalty': ramp_down_penalty,
            'if_max_penalty': if_max_penalty,
            'if_min_penalty': if_min_penalty,
            'br_max_penalty': br_max_penalty,
            'br_min_penalty': br_min_penalty
        }

        # Combine results
        results = {**vars, **slack_vars, **costs}

        return results

    def show_model_dim(self):
        """
        Show model dimensions.
        """
        print('Number of buses: {}'.format(self.NB))
        print('Number of generators: {}'.format(self.NG))
        print('Number of branches: {}'.format(self.NBR))
        print('Number of time periods: {}'.format(self.NT))

        num_vars = self.model.nvariables()
        print('Number of variables: {}'.format(num_vars))

        num_constraints = self.model.nconstraints()
        print('Number of constraints: {}'.format(num_constraints))

    def get_last_gen(self, model_multi_opf):
        """
        Get generator power generation at the last simulation.
        Used to create initial condition for the next simulation.
        """
        # Get dimensions of the last simulation
        NT = len(model_multi_opf.PG_index_0)
        NG = len(model_multi_opf.PG_index_1)
        results_pg = np.array(model_multi_opf.PG[:, :]()).reshape(NT, NG) * self.baseMVA
        last_gen = results_pg[-1, :]

        return last_gen

"""
Class for running the NYGrid model.

Known Issues/Wishlist:
1. Better dc line model
2. Better documentation
3. Check dim of start/end datetime and load profile

"""
import logging
import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pypower.api as pp
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.opt.results.results_ import SolverResults

from nygrid.optimizer import Optimizer
from nygrid.ppc_idx import *
from nygrid.utlis import format_date

from typing import Optional, Union, Dict, Tuple


def check_status(results: SolverResults) -> bool:
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
        logging.info("The problem is feasible and optimal!")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        raise RuntimeError("The problem is infeasible!")
    else:
        logging.error(str(results.solver))
        raise RuntimeError("Something else is wrong!")
    return True


def read_grid_data(grid_data_dir: Union[str, os.PathLike]) -> Dict[str, pd.DataFrame]:
    """
    Read grid data from csv files.

    Parameters
    ----------
    grid_data_dir: str
        Path to the grid data directory.

    Returns
    -------
    grid_data: dict
        Dictionary of grid data.
        Keys: bus_prop, gen_prop, gen_fuel, gencost_prop, branch_prop, if_lim_prop, if_map_prop, storage_prop
        Values: pandas.DataFrame
    """

    # Read bus properties
    filename = os.path.join(grid_data_dir, 'bus_prop.csv')
    if os.path.exists(filename):
        bus_prop = pd.read_csv(filename, index_col=0)
    else:
        raise ValueError('Bus properties file does not exist.')

    # Read generator properties
    filename = os.path.join(grid_data_dir, 'gen_prop.csv')
    if os.path.exists(filename):
        gen_prop = pd.read_csv(filename, index_col=0)
    else:
        raise ValueError('Generator properties file does not exist.')

    # Read generator fuel type
    filename = os.path.join(grid_data_dir, 'genfuel_prop.csv')
    if os.path.exists(filename):
        gen_fuel = pd.read_csv(filename, index_col=0)
    else:
        raise ValueError('Generator fuel type file does not exist.')

    # Read generator cost properties
    filename = os.path.join(grid_data_dir, 'gencost_prop.csv')
    if os.path.exists(filename):
        gencost_prop = pd.read_csv(filename, index_col=0)
    else:
        raise ValueError('Generator cost properties file does not exist.')

    # Read AC line properties
    filename = os.path.join(grid_data_dir, 'branch_prop.csv')
    if os.path.exists(filename):
        branch_prop = pd.read_csv(filename, index_col=0)
        # Replace default 0 (unlimited) with 9999
        branch_prop.loc[branch_prop['RATE_A'] == 0, 'RATE_A'] = 9999
    else:
        raise ValueError('AC line properties file does not exist.')

    # Read interface properties
    filename = os.path.join(grid_data_dir, 'if_lims_prop.csv')
    if os.path.exists(filename):
        if_lim_prop = pd.read_csv(filename, index_col=0)
    else:
        raise ValueError('Interface limit properties file does not exist.')

    # Read interface mapping
    filename = os.path.join(grid_data_dir, 'if_map_prop.csv')
    if os.path.exists(filename):
        if_map_prop = pd.read_csv(filename, index_col=0)
    else:
        raise ValueError('Interface mapping file does not exist.')

    # Read storage properties (Optional)
    filename = os.path.join(grid_data_dir, 'esr_prop.csv')
    if os.path.exists(filename):
        storage_prop = pd.read_csv(filename, index_col=0)
    else:
        Warning('No storage properties are provided.')
        storage_prop = None

    # Read DC line properties (Optional)
    filename = os.path.join(grid_data_dir, 'dcline_prop.csv')
    if os.path.isfile(filename):
        dcline_prop = pd.read_csv(filename, index_col=0)
    else:
        Warning('No DC line properties are provided.')
        dcline_prop = None

    grid_data = {
        'bus_prop': bus_prop,
        'gen_prop': gen_prop,
        'gen_fuel': gen_fuel,
        'gencost_prop': gencost_prop,
        'branch_prop': branch_prop,
        'if_lim_prop': if_lim_prop,
        'if_map_prop': if_map_prop,
        'storage_prop': storage_prop,
        'dcline_prop': dcline_prop
    }

    return grid_data


def convert_dcline_2_gen(ppc: Dict[str, np.ndarray],
                         dcline_prop: Optional[Union[np.ndarray, pd.DataFrame]] = None
                         ) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Convert DC lines to generators and add their parameters in the PyPower matrices.
    For each DC line, add two injectors: one at FROM bus and another at TO bus.
    The injection of them are linked in the optimization.

    Parameters
    ----------
    ppc: dict
        PyPower case dictionary.
    dcline_prop: numpy.ndarray or pandas.DataFrame
        DC line properties.

    Returns
    -------
    ppc_dc: dict
        PyPower case dictionary with DC lines converted to generators.
    num_dcline: int
        Number of DC lines.
        ppc (dict): PyPower case dictionary.
    """

    # Get PyPower case information
    ppc_dc = ppc.copy()

    if dcline_prop is not None and dcline_prop.size > 0:
        dcline = dcline_prop
    else:
        dcline = ppc_dc['dcline']
        Warning('No DC line properties are provided. Use default values from the PyPower case.')

    if isinstance(dcline, pd.DataFrame):
        dcline = dcline.to_numpy()
    elif isinstance(dcline, np.ndarray):
        pass
    else:
        raise ValueError('DC line properties must be a numpy.ndarray or pandas.DataFrame.')

    # Set gen parameters of the DC line converted generators
    num_dcline = dcline.shape[0]
    dcline_gen = np.zeros((num_dcline * 2, 21))  # One at from bus, one at to bus
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
    dcline_gen[:, MBASE] = np.ones(num_dcline * 2) * ppc['baseMVA']
    dcline_gen[:, GEN_STATUS] = np.concatenate([dcline[:, DC_BR_STATUS],
                                                dcline[:, DC_BR_STATUS]])
    dcline_gen[:, PMAX] = np.concatenate([dcline[:, DC_PMAX],
                                          dcline[:, DC_PMAX]])
    dcline_gen[:, PMIN] = np.concatenate([dcline[:, DC_PMIN],
                                          dcline[:, DC_PMIN]])
    dcline_gen[:, RAMP_AGC] = np.ones(num_dcline * 2) * 1e6  # Unlimited ramp rate
    dcline_gen[:, RAMP_10] = np.ones(num_dcline * 2) * 1e6  # Unlimited ramp rate
    dcline_gen[:, RAMP_30] = np.ones(num_dcline * 2) * 1e6  # Unlimited ramp rate
    # Add the DC line converted generators to the gen matrix
    ppc_dc['gen'] = np.concatenate([ppc['gen'], dcline_gen])

    # Set gencost parameters of the DC line converted generators
    dcline_gencost = np.zeros((num_dcline * 2, 6))
    dcline_gencost[:, MODEL] = np.ones(num_dcline * 2) * POLYNOMIAL
    dcline_gencost[:, NCOST] = np.ones(num_dcline * 2) * 2
    # Add the DC line converted generators to the gencost matrix
    ppc_dc['gencost'] = np.concatenate([ppc['gencost'], dcline_gencost])

    # Add the DC line converted generators to the genfuel list
    dcline_genfuel = np.array(['DC Line From'] * num_dcline
                              + ['DC Line To'] * num_dcline).reshape(2 * num_dcline, 1)
    ppc_dc['genfuel'] = np.concatenate([ppc['genfuel'], dcline_genfuel])

    return ppc_dc, num_dcline


def convert_esr_2_gen(ppc: Dict[str, np.ndarray],
                      esr_prop: Optional[Union[np.ndarray, pd.DataFrame]] = None
                      ) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Convert ESR to generators and add their parameters in the PyPower matrices.
    For each ESR, add one injector to represent the combined injection of the ESR.
    Positive injection is discharging and negative injection is charging.

    Parameters
    ----------
    ppc: dict
        PyPower case dictionary.
    esr_prop: numpy.ndarray or pandas.DataFrame
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

    if esr_prop is None or esr_prop.size == 0:
        Warning('No ESR properties are provided.')
        return ppc_esr, 0

    if isinstance(esr_prop, pd.DataFrame):
        esr_prop = esr_prop.to_numpy()
    elif isinstance(esr_prop, np.ndarray):
        pass
    else:
        raise ValueError('ESR properties must be a numpy.ndarray or pandas.DataFrame.')

    # Set gen parameters of the ESR converted generators
    num_esr = esr_prop.shape[0]
    esr_gen = np.zeros((num_esr, 21))  # One for charging, one for discharging
    esr_gen[:, GEN_BUS] = np.array(esr_prop[:, ESR_BUS])
    esr_gen[:, PG] = np.zeros(num_esr)
    esr_gen[:, QG] = np.zeros(num_esr)
    esr_gen[:, QMAX] = np.zeros(num_esr)
    esr_gen[:, QMIN] = np.zeros(num_esr)
    esr_gen[:, VG] = np.ones(num_esr)
    esr_gen[:, MBASE] = np.ones(num_esr) * ppc['baseMVA']
    esr_gen[:, GEN_STATUS] = np.ones(num_esr)
    esr_gen[:, PMAX] = np.array(esr_prop[:, ESR_DIS_MAX])
    esr_gen[:, PMIN] = np.array(-1 * esr_prop[:, ESR_CRG_MAX])
    esr_gen[:, RAMP_AGC] = np.ones(num_esr) * 1e6  # Unlimited ramp rate
    esr_gen[:, RAMP_10] = np.ones(num_esr) * 1e6  # Unlimited ramp rate
    esr_gen[:, RAMP_30] = np.ones(num_esr) * 1e6  # Unlimited ramp rate
    # Add the ESR converted generators to the gen matrix
    ppc_esr['gen'] = np.concatenate([ppc['gen'], esr_gen])

    # Set gencost parameters of the ESR converted generators
    esr_gencost = np.zeros((num_esr, 6))
    esr_gencost[:, MODEL] = np.ones(num_esr) * POLYNOMIAL
    esr_gencost[:, NCOST] = np.ones(num_esr) * 2
    # Add the ESR converted generators to the gencost matrix
    ppc_esr['gencost'] = np.concatenate([ppc['gencost'], esr_gencost])

    # Add the ESR converted generators to the genfuel list
    esr_genfuel = np.array(['ESR'] * num_esr).reshape(num_esr, 1)
    ppc_esr['genfuel'] = np.concatenate([ppc['genfuel'], esr_genfuel])

    return ppc_esr, num_esr


def convert_vre_2_gen(ppc: Dict[str, np.ndarray],
                      vre_prop: Optional[Union[np.ndarray, pd.DataFrame]] = None
                      ) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Convert renewable generators to generators and add their parameters in the PyPower matrices.

    Parameters
    ----------
    ppc: dict
        PyPower case dictionary.
    vre_prop: numpy.ndarray or pandas.DataFrame
        VRE properties.

    Returns
    -------
    ppc_vre: dict
        PyPower case dictionary with VRE converted to generators.
    num_vre: int
        Number of VRE.
    """

    # Get PyPower case information
    ppc_vre = ppc.copy()

    if vre_prop is None or vre_prop.size == 0:
        Warning('No ESR properties are provided.')
        return ppc_vre, 0

    if isinstance(vre_prop, pd.DataFrame):
        vre_prop = vre_prop.to_numpy()
    elif isinstance(vre_prop, np.ndarray):
        pass
    else:
        raise ValueError('VRE properties must be a numpy.ndarray or pandas.DataFrame.')

    # Set gen parameters of the VRE converted generators
    num_vre = vre_prop.shape[0]
    vre_gen = np.zeros((num_vre, 21))  # One for charging, one for discharging
    vre_gen[:, GEN_BUS] = np.array(vre_prop[:, VRE_BUS])
    vre_gen[:, PG] = np.zeros(num_vre)
    vre_gen[:, QG] = np.zeros(num_vre)
    vre_gen[:, QMAX] = np.zeros(num_vre)
    vre_gen[:, QMIN] = np.zeros(num_vre)
    vre_gen[:, VG] = np.ones(num_vre)
    vre_gen[:, MBASE] = np.ones(num_vre) * ppc['baseMVA']
    vre_gen[:, GEN_STATUS] = np.ones(num_vre)
    vre_gen[:, PMAX] = np.array(vre_prop[:, VRE_PMAX])
    vre_gen[:, PMIN] = np.array(vre_prop[:, VRE_PMIN])
    vre_gen[:, RAMP_AGC] = np.ones(num_vre) * 1e6  # Unlimited ramp rate
    vre_gen[:, RAMP_10] = np.ones(num_vre) * 1e6  # Unlimited ramp rate
    vre_gen[:, RAMP_30] = np.ones(num_vre) * 1e6  # Unlimited ramp rate
    # Add the ESR converted generators to the gen matrix
    ppc_vre['gen'] = np.concatenate([ppc['gen'], vre_gen])

    # Set gencost parameters of the VRE converted generators
    vre_gencost = np.zeros((num_vre, 6))
    vre_gencost[:, MODEL] = np.ones(num_vre) * POLYNOMIAL
    vre_gencost[:, NCOST] = np.ones(num_vre) * 2
    # Add the ESR converted generators to the gencost matrix
    ppc_vre['gencost'] = np.concatenate([ppc['gencost'], vre_gencost])

    # Add the VRE converted generators to the genfuel list
    vre_genfuel = np.array(vre_prop[:, VRE_TYPE]).reshape(num_vre, 1)
    ppc_vre['genfuel'] = np.concatenate([ppc['genfuel'], vre_genfuel])

    return ppc_vre, num_vre


class NYGrid:
    """
    Class for running the NYGrid model.

    """

    def __init__(self,
                 grid_data_dir: Union[str, os.PathLike],
                 start_datetime: Union[str, pd.Timestamp],
                 end_datetime: Union[str, pd.Timestamp],
                 dcline_prop: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 esr_prop: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 vre_prop: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 verbose: bool = False) -> None:
        """
        Initialize the NYGrid model.

        Parameters
        ----------
        grid_data_dir: str
            Path to the grid data directory.
        start_datetime: str
            Start datetime of the simulation.
        end_datetime: str
            End datetime of the simulation.
        verbose: bool
            If True, print out the information of the simulation.

        Returns
        -------
        NYGrid: nygrid.nygrid.NYGrid
            NY Grid object.
        """

        # %% Load PyPower case
        # self.ppc = pp.loadcase(ppc_filename)

        # %% Set the start and end datetime of the simulation
        self.verbose = verbose

        # Format the forecast start/end and determine the total time.
        if isinstance(start_datetime, pd.Timestamp):
            self.start_datetime = start_datetime
        elif isinstance(start_datetime, str):
            self.start_datetime = format_date(start_datetime)

        if isinstance(end_datetime, pd.Timestamp):
            self.end_datetime = end_datetime
        elif isinstance(end_datetime, str):
            self.end_datetime = format_date(end_datetime)

        self.delta_t = self.end_datetime - self.start_datetime
        self.timestamp_list = pd.date_range(self.start_datetime, self.end_datetime, freq='1H')
        self.NT = len(self.timestamp_list)

        if self.verbose:
            logging.info('Initializing NYGrid run...')
            logging.info(f'NYGrid run starting on: {self.start_datetime}')
            logging.info(f'NYGrid run ending on: {self.end_datetime}')
            logging.info(f'NYGrid run duration: {self.delta_t}')

        # %% Read grid data
        self.grid_data = read_grid_data(grid_data_dir)

        # %% Create PyPower case
        self.ppc = dict()
        self.ppc['baseMVA'] = 100
        self.ppc['version'] = '2'
        self.ppc['bus'] = self.grid_data['bus_prop'].drop(columns=['BUS_ZONE']).to_numpy()
        self.ppc['gen'] = self.grid_data['gen_prop'].drop(columns=['GEN_NAME', 'GEN_ZONE', 'GEN_FUEL']).to_numpy()
        self.ppc['genfuel'] = self.grid_data['gen_fuel'].to_numpy()
        self.ppc['gencost'] = self.grid_data['gencost_prop'].to_numpy()
        self.ppc['branch'] = self.grid_data['branch_prop'].drop(columns=['FROM_ZONE', 'TO_ZONE']).to_numpy()
        self.ppc['dcline'] = self.grid_data['dcline_prop'].drop(columns=['FROM_ZONE', 'TO_ZONE']).to_numpy()
        self.ppc['if'] = {'lims': self.grid_data['if_lim_prop'].drop(columns=['FROM_ZONE', 'TO_ZONE']).to_numpy(),
                          'map': self.grid_data['if_map_prop'].drop(columns=['BR_IDX', 'BR_DIR']).to_numpy()}

        # Convert DC line to generators and add to gen matrix
        self.ppc_dc, self.NDCL = convert_dcline_2_gen(self.ppc, dcline_prop)

        # Convert ESR to generators and add to gen matrix
        self.ppc_dc_esr, self.NESR = convert_esr_2_gen(self.ppc_dc, esr_prop)

        # Convert renewable generators to generators and add to gen matrix
        self.ppc_dc_esr_vre, self.NVRE = convert_vre_2_gen(self.ppc_dc_esr, vre_prop)

        # Convert to internal indexing
        self.ppc_int = pp.ext2int(self.ppc_dc_esr_vre)

        self.baseMVA = self.ppc_int['baseMVA']
        self.bus = self.ppc_int['bus']
        self.gen = self.ppc_int['gen']
        self.branch = self.ppc_int['branch']
        self.gencost = self.ppc_int['gencost']

        # Generator info
        self.gen_bus = self.gen[:, GEN_BUS].astype(int)  # what buses are they at?

        # Build B matrices and phase shift injections
        B, Bf, _, _ = pp.makeBdc(self.baseMVA, self.bus, self.branch)
        self.B = B.todense
        self.Bf = Bf.todense()

        # Linear shift factor
        self.PTDF = pp.makePTDF(self.baseMVA, self.bus, self.branch)

        # Problem dimensions
        self.NG = self.gen.shape[0]  # Number of generators
        self.NB = self.bus.shape[0]  # Number of buses
        self.NBR = self.branch.shape[0]  # Number of lines
        self.NL = np.size(self.bus[:, PD])  # Number of loads

        # Get mapping from generator to bus
        self.gen_map = np.zeros((self.NB, self.NG))
        self.gen_map[self.gen_bus, range(self.NG)] = 1

        # Get index of existing generators (not DC line or ESR converted)
        self.gen_i2e = self.ppc_int['order']['gen']['i2e']
        self.gen_idx_non_cvt = self.gen_i2e[:self.NG - self.NDCL * 2 - self.NESR - self.NVRE]

        # Get index of DC line converted generators in internal indexing
        self.dcline_idx_f = self.gen_i2e[self.NG - self.NDCL * 2 - self.NESR - self.NVRE:
                                         self.NG - self.NDCL - self.NESR - self.NVRE]
        self.dcline_idx_t = self.gen_i2e[self.NG - self.NDCL - self.NESR - self.NVRE:
                                         self.NG - self.NESR - self.NVRE]

        # Get index of ESR converted generators in internal indexing
        self.esr_idx = self.gen_i2e[self.NG - self.NESR - self.NVRE: self.NG - self.NVRE]

        # Get index of VRE converted generators in internal indexing
        self.vre_idx = self.gen_i2e[self.NG - self.NVRE: self.NG]

        # Get mapping from load to bus
        self.load_map = np.zeros((self.NB, self.NL))
        self.load_bus = np.nonzero(self.bus[:, PD])[0]
        for i in range(len(self.load_bus)):
            self.load_map[self.load_bus[i], i] = 1

        # Line flow limit in p.u.
        self.br_max = self.branch[:, RATE_A] / self.baseMVA
        # Replace default value 0 (Unlimited) to 999.99
        self.br_max[self.br_max == 0] = 999.99
        self.br_min = - self.br_max

        # Get interface limit information
        self.if_map = self.ppc_int['if']['map']
        self.if_lims = self.ppc_int['if']['lims'].astype(float)
        self.if_lims[:, 1:] = self.if_lims[:, 1:] / self.baseMVA
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
        # self.ramp_down = np.min([self.gen_max, self.ramp_up], axis=0)
        self.ramp_down = self.ramp_up

        # Linear cost intercept coefficients in p.u.
        self.gencost_0 = np.ones((self.NT, self.NG)) * self.gencost[:, COST + 1]

        # Linear cost slope coefficients in p.u.
        self.gencost_1 = np.ones((self.NT, self.NG)) * self.gencost[:, COST] * self.baseMVA

        # Convert load to p.u.
        self.load_pu = np.ones((self.NT, self.NL)) * self.bus[:, PD] / self.baseMVA

        # Generator initial condition
        self.gen_init = None

        # Add ESR properties
        if esr_prop is not None and esr_prop.size > 0:

            if isinstance(esr_prop, pd.DataFrame):
                esr_prop = esr_prop.to_numpy()
            elif isinstance(esr_prop, np.ndarray):
                pass
            else:
                raise ValueError('ESR properties must be a numpy.ndarray or pandas.DataFrame.')

            # ESR charging power upper limit in p.u.
            self.esr_crg_max = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_CRG_MAX] / self.baseMVA

            # ESR discharging power upper limit in p.u.
            self.esr_dis_max = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_DIS_MAX] / self.baseMVA

            # ESR charging efficiency
            self.esr_crg_eff = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_CRG_EFF]

            # ESR discharging efficiency
            self.esr_dis_eff = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_DIS_EFF]

            # ESR SOC upper limit in p.u.
            self.esr_soc_max = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_SOC_MAX] / self.baseMVA

            # ESR SOC lower limit in p.u.
            self.esr_soc_min = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_SOC_MIN] / self.baseMVA

            # ESR SOC initial condition in p.u.
            self.esr_init = np.ones(self.NESR) * esr_prop[:, ESR_SOC_INI] / self.baseMVA

            # ESR SOC target condition in p.u.
            self.esr_target = np.ones(self.NESR) * esr_prop[:, ESR_SOC_TGT] / self.baseMVA

            # ESR charging cost
            self.esrcost_crg = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_CRG_COST] * self.baseMVA

            # ESR discharging cost
            self.esrcost_dis = np.ones((self.NT, self.NESR)) * esr_prop[:, ESR_DIS_COST] * self.baseMVA

        # %% Create Pyomo model
        self.model = pyo.ConcreteModel(name='multi-period DC OPF')

        # %% Penalty parameters and default values
        # ED module
        self.NoPowerBalanceViolation = False
        self.NoRampViolation = False
        self.PenaltyForOverGeneration = 1_500  # $/MWh
        self.PenaltyForLoadShed = 20_000  # $/MWh
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
        self.PenaltyForBranchMwViolation = 10_000  # $/MWh
        self.PenaltyForInterfaceMWViolation = 10_000  # $/MWh
        self.MaxPhaseAngleDifference = 1.5  # Radians, Not used
        # ES module
        self.PenaltyForEnergyStorageViolation = 8_000  # $/MWh, Not used

        # Model options
        # Use Power Transfer Distribution Factors (PTDF) for linear shift factor
        self.UsePTDF = True
        self.solver = 'gurobi'

    def set_options(self, options: Dict[str, Union[int, float]]) -> None:
        """
        Set solver options and penalty parameters.

        Parameters
        ----------
        options: dict
            Solver options and penalty parameters.

        Returns
        -------
        None
        """

        for key, value in options.items():
            setattr(self, key, value)
            if self.verbose:
                logging.info(f'Set {key} to {value} ...')

    def set_load_sch(self, load_sch: pd.DataFrame) -> None:
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

    def set_gen_mw_sch(self, gen_mw_sch: pd.DataFrame) -> None:
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
            self.gen_hist[:, self.gen_idx_non_cvt] = gen_mw_sch / self.baseMVA
            # HVDC Proxy generators: Use default values from the PyPower case
            self.gen_hist[:, self.dcline_idx_f] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dcline_idx_f, PG] / self.baseMVA
            self.gen_hist[:, self.dcline_idx_t] = np.ones(
                (self.NT, self.NDCL)) * self.gen[self.dcline_idx_t, PG] / self.baseMVA
        else:
            raise ValueError('No generation profile is provided.')

    def set_gen_max_sch(self, gen_max_sch: pd.DataFrame) -> None:
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
            self.gen_max[:, self.gen_idx_non_cvt] = gen_max_sch / self.baseMVA

        else:
            raise ValueError('No generation capacity profile is provided.')

    def set_vre_max_sch(self, vre_max_sch: pd.DataFrame) -> None:
        """
        Set VRE upper operating limit data from generation capacity profile.

        Parameters
        ----------
        vre_max_sch: pandas.DataFrame
            VRE upper operating limit profile of thermal generators.

        Returns
        -------
        None
        """

        # Slice the generator profile to the simulation period
        vre_max_sch = vre_max_sch[self.start_datetime:self.end_datetime].to_numpy()

        # Generator upper operating limit in p.u.
        if vre_max_sch is not None and vre_max_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.gen_max[:, self.vre_idx] = vre_max_sch / self.baseMVA

        else:
            raise ValueError('No VRE generation capacity profile is provided.')

    def set_gen_min_sch(self, gen_min_sch: pd.DataFrame) -> None:
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
            self.gen_min[:, self.gen_idx_non_cvt] = gen_min_sch / self.baseMVA

        else:
            raise ValueError('No generation capacity profile is provided.')

    def set_gen_ramp_sch(self, gen_ramp_sch: pd.DataFrame,
                         interval: str = '30min') -> None:
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

            # Convert default value 0 (Unlimited) to 1e6
            gen_ramp_sch[gen_ramp_sch == 0] = 1e6
        else:
            raise ValueError('Only support 30min ramp rate profile.')

        # Generator ramp rate limit in p.u./hour
        if gen_ramp_sch is not None and gen_ramp_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.ramp_up[:, self.gen_idx_non_cvt] = gen_ramp_sch / self.baseMVA

        else:
            raise ValueError('No ramp rate profile is provided.')

        # Downward ramp rate is the minimum of the upward ramp rate and the maximum generation limit
        # self.ramp_down = np.min([self.gen_max, self.ramp_up], axis=0)
        self.ramp_down = self.ramp_up

    def set_gen_cost_sch(self, gen_cost0_sch: pd.DataFrame,
                         gen_cost1_sch: pd.DataFrame) -> None:
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
            self.gencost_0[:, self.gen_idx_non_cvt] = gen_cost0_sch

        else:
            raise ValueError('No generation cost profile is provided.')

        # Linear cost slope coefficients in p.u.
        if gen_cost1_sch is not None and gen_cost1_sch.size > 0:
            # Thermal generators: Use user-defined time series schedule
            self.gencost_1[:, self.gen_idx_non_cvt] = gen_cost1_sch * self.baseMVA

        else:
            raise ValueError('No generation cost profile is provided.')

    def relax_external_branch_lim(self):
        """
        Relax external branch flow limit to 999.99.

        Returns
        -------
        None
        """

        self.br_max[self.br_max != 999.99] = 999.99
        self.br_min[self.br_min != -999.99] = -999.99

    def set_gen_init_data(self, gen_init: Optional[np.ndarray]) -> None:
        """
        Get generator initial condition.

        Parameters
        ----------
            gen_init (numpy.ndarray): A 1-d array of generator initial condition

        """

        if gen_init is not None and gen_init.size > 0:
            # Convert to internal generator indexing
            gen_init = pd.DataFrame(gen_init, index=self.gen_i2e).sort_index().to_numpy().squeeze()
            self.gen_init = gen_init / self.baseMVA
        else:
            Warning('No generator initial condition is provided.')

    def check_input_dim(self) -> None:
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

    def create_dc_opf(self) -> None:
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

        # Add ES variables if there are ESRs
        if self.NESR > 0:
            optimizer.add_vars_es()

        # Add dual variables
        optimizer.add_vars_dual()

        # Add constraints
        optimizer.add_constrs_ed()
        optimizer.add_constrs_pf()

        # Add ES constraints if there are ESRs
        if self.NESR > 0:
            optimizer.add_constrs_es()

        # Add objective
        optimizer.add_obj()

        self.model = optimizer.model

    def solve_dc_opf(self, solver_options: Optional[Dict[str, Union[int, float]]] = None) -> None:
        """
        Solve a multi-period DC OPF problem.

        Parameters
        ----------
        solver_options: dict
            Solver options.

        Returns
        -------
        None
        """

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
            logging.info(f"Objective function value: {self.model.obj():.3e}")

    def get_results_dc_opf(self) -> Dict[str, pd.DataFrame]:
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
        variables = dict()

        # Power generation
        results_pg = np.array(self.model.PG[:, :]()).reshape(self.NT, self.NG) * self.baseMVA
        gen_order = self.ppc_int['order']['gen']['e2i']
        results_pg = pd.DataFrame(results_pg, index=self.timestamp_list,
                                  columns=gen_order).sort_index(axis=1)
        variables['PG'] = results_pg

        # Branch power flow
        branch_pf = np.array(self.model.PF[:, :]()).reshape(self.NT, self.NBR) * self.baseMVA
        results_pf = pd.DataFrame(branch_pf, index=self.timestamp_list)
        variables['PF'] = results_pf

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
        variables['IF'] = results_if

        if not self.UsePTDF:
            # Bus phase angle
            results_va = np.array(self.model.VA[:, :]()).reshape(self.NT, self.NB) * 180 / np.pi
            # Just to compare with PyPower
            results_va = results_va - 73.4282
            # Convert negative numbers to 0-360
            results_va = np.where(results_va < 0, results_va + 360, results_va)
            results_va = pd.DataFrame(results_va, index=self.timestamp_list)
            variables['VA'] = results_va

            # Bus locational marginal price (LMP)
            results_lmp = np.zeros((self.NT, self.NB))
            for t in range(self.NT):
                for n in range(self.NB):
                    results_lmp[t, n] = np.abs(self.model.dual[self.model.c_pf[t, n]]) / self.baseMVA
            results_lmp = pd.DataFrame(results_lmp, index=self.timestamp_list)
            variables['LMP'] = results_lmp

        else:
            # Bus power injection
            results_pbus = np.array(self.model.PBUS[:, :]()).reshape(self.NT, self.NB) * self.baseMVA
            results_pbus = pd.DataFrame(results_pbus, index=self.timestamp_list)
            variables['PBUS'] = results_pbus

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
            variables['LMP'] = results_lmp

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

        total_cost = (gen_cost + over_gen_penalty + load_shed_penalty
                      + ramp_up_penalty + ramp_down_penalty
                      + if_max_penalty + if_min_penalty
                      + br_max_penalty + br_min_penalty)

        # TODO: Add ES cost and penalty

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
        results = {**variables, **slack_vars, **costs}

        return results

    def show_model_dim(self) -> None:
        """
        Show model dimensions.
        """
        logging.info('Number of buses: {}'.format(self.NB))
        logging.info('Number of generators: {}'.format(self.NG))
        logging.info('Number of branches: {}'.format(self.NBR))
        logging.info('Number of time periods: {}'.format(self.NT))

        num_vars = self.model.nvariables()
        logging.info('Number of variables: {}'.format(num_vars))

        num_constraints = self.model.nconstraints()
        logging.info('Number of constraints: {}'.format(num_constraints))

    def get_last_gen(self, model_multi_opf: pyo.ConcreteModel) -> np.ndarray:
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

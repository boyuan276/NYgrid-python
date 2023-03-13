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

def single_opf(B, Bf, gen_map, load_map, load, gencost_0, gencost_1, dc_idx_f, dc_idx_t,
                gen_min, gen_max, br_min, br_max, if_map, if_lims):
    '''
    Single-period OPF problem.

    Parameters:
        A tuple of OPF network parameters and constraints.
    
    Returns:
        model (pyomo.core.base.PyomoModel.ConcreteModel): Pyomo model for single-period OPF problem.
    '''
    # Get problem dimensions
    NB, NG = gen_map.shape
    NL = load.shape[0]
    NBR = br_min.shape[0]

    model = ConcreteModel(name='single-period OPF')

    # Define variables
    model.PG = Var(range(NG), within=Reals,initialize=1)  # Real power generation
    model.Va = Var(range(NB), within=Reals, initialize=0,
                   bounds=(-2*np.pi, 2*np.pi))  # Bus voltage angle
    model.PF = Var(range(NBR), within=Reals, initialize=0) # Branch power flow
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
    for br in range(NBR):
        model.c_br_flow.add(model.PF[br] == sum(Bf[br, b]*model.Va[b] for b in range(NB)))
        model.c_br_max.add(model.PF[br] <= br_max[br])
        model.c_br_min.add(model.PF[br] >= br_min[br])
    
    # Generation capacity limit
    for g in range(NG):
        model.c_gen_max.add(model.PG[g] <= gen_max[g])
        model.c_gen_min.add(model.PG[g] >= gen_min[g])
        
    # DC power flow constraint
    for b in range(NB):
        model.c_pf.add(sum(gen_map[b, g]*model.PG[g] for g in range(NG)) - sum(load_map[b, l]*load[l] for l in range(NL))
                        == sum(B[b, b_]*model.Va[b_] for b_ in range(NB)))
    
    # DC line power balance constraint
    for idx_f, idx_t in zip(dc_idx_f, dc_idx_t):
        model.c_dcline.add(model.PG[idx_f] == -model.PG[idx_t])
    
    # Interface flow limits
    for n in range(len(if_lims)):
        if_id, if_lims_min, if_lims_max = if_lims[n, :]
        br_dir_idx = if_map[(if_map[:,0] == int(if_id)), 1]
        br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
        model.c_if_max.add(if_lims_max >= sum(br_dir[i]*model.PF[br_idx[i]] for i in range(len(br_idx))))
        model.c_if_min.add(if_lims_min <= sum(br_dir[i]*model.PF[br_idx[i]] for i in range(len(br_idx))))

    def cost(model, gencost_0, gencost_1):

        cost = sum(gencost_0[i] for i in range(NG)) \
            + sum(gencost_1[i]*model.PG[i] for i in range(NG))

        return cost

    model.obj = Objective(expr=cost(model, gencost_0, gencost_1), sense=minimize)

    return model

def single_opf_new(B, Bf, gen_map, load_map, load, gencost_0, gencost_1, dc_idx_f, dc_idx_t,
                gen_min, gen_max, br_min, br_max, if_map, if_lims):

    # Get problem dimensions
    NB, NG = gen_map.shape
    NL = load.shape[0]
    NBR = br_min.shape[0]
    NI = if_lims.shape[0]
    NDC = dc_idx_f.shape[0]
    
    model = ConcreteModel(name='single-period OPF')
    
    # Define sets
    model.BUS = Set(initialize=range(NG), doc="Generator")
    model.BRCH = Set(initialize=range(NBR), doc="Branch")
    model.GEN = Set(initialize=range(NG), doc="Bus")
    model.LOAD = Set(initialize=range(NL), doc="Load")
    model.IFACE = Set(initialize=range(NI), doc="Interface")
    model.DC = Set(initialize=range(NDC), doc="DC line")

    # Define parameters
    def gen_map_init(model, b, g):
        return gen_map[b, g]
    model.gen_map = Param(model.BUS, model.GEN, initialize=gen_map_init, 
                            doc="Mapping from generator to bus")
    
    def load_map_init(model, b, l):
        return load_map[b, l]
    model.load_map = Param(model.BUs, model.LOAD, initialize=load_map_init,
                            doc="Mapping from load to bus")

    def load_init(model, l):
        return load[l]
    model.load = Param(model.LOAD, initialize=load_init,
                        doc="Load")
    
    def B_init(model, b, b_):
        return B[b, b_]
    model.B = Param(model.BUS, model.BUS, initialize=B_init,
                    doc="Power injection impedance matrix")

    def Bf_init(model, br, b):
        return Bf[br, b]
    model.Bf = Param(model.BRCH, model.BUS, initialize=Bf_init,
                    doc="Line flow impedance matrix")

    def br_max_init(model, br):
        return br_max(br)
    model.br_max = Param(model.BRCH, initialize=br_max_init,
                        doc="Branch flow upper limit")
    
    def br_min_init(model, br):
        return br_min(br)
    model.br_min = Param(model.BRCH, initialize=br_min_init,
                        doc="Branch flow lower limit")
    
    def gen_max_init(model, g):
        return gen_max(g)
    model.gen_max = Param(model.GEN, initialize=gen_max_init,
                        doc="Generator upper limit")
    
    def gen_min_init(model, g):
        return gen_min(g)
    model.gen_min = Param(model.GEN, initialize=gen_min_init,
                        doc="Generator lower limit")
    
    def if_br_dir_init(model, i):
        br_dir_idx = model.if_map[(model.if_map[:, 0] == i), 1]
        return np.sign(br_dir_idx)
    model.if_br_dir = Param(model.IFACE, initialize=if_br_dir_init,
                            doc="Branch direction of interface limit")

    def if_br_idx_init(model, i):
        br_dir_idx = model.if_map[(model.if_map[:, 0] == i), 1]
        return np.abs(br_dir_idx).astype(int)
    model.if_br_idx = Param(model.IFACE, initialize=if_br_idx_init,
                            doc="Branch indices of interface limit")

    def if_lim_max_init(model, i):
        return if_lims[i, 2]
    model.if_lim_max = Param(model.IFACE, initialize=if_lim_max_init,
                            doc="Interface flow lower limit")
    
    def if_lim_min_init(model, i):
        return if_lims[i, 1]
    model.if_lim_min = Param(model.IFACE, initialize=if_lim_min_init,
                            doc="Interface flow lower limit")
    
    def dc_idx_f_init(model, dc):
        return dc_idx_f[dc]
    model.dc_idx_f = Param(model.DC, initialize=dc_idx_f_init,
                            doc="DC line source virtual generator index")
    
    def dc_idx_t_init(model, dc):
        return dc_idx_f[dc]
    model.dc_idx_t = Param(model.DC, initialize=dc_idx_t_init,
                            doc="DC line sink virtual generator index")
    
    def gencost_0_init(model, g):
        return gencost_0[g]
    model.gencost_0 = Param(model.GEN, initialize=gencost_0_init,
                            doc="Intercept of generator cost function")
    
    def gencost_1_init(model, g):
        return gencost_1[g]
    model.gencost_1 = Param(model.GEN, initialize=gencost_1_init,
                            doc="Slope of generator cost function")

    # Define variables
    model.pg = Var(model.GEN, within=Reals, initialize=1,
                    doc="Generator dispatch")
    model.va = Var(model.BUS, within=Reals, initialize=0, 
                    bounds=(-2*np.pi, 2*np.pi), doc="Bus phase angle")
    model.pf = Var(model.BRCH, within=Reals, initialize=0, 
                    doc="Branch power flow")

    # Define constraints
    # Generator operating limits
    def gen_max_rule(model, g):
        return model.pg[g] <= model.gen_max[g]
    model.c_gen_max = Constraint(model.GEN, rule=gen_max_rule)

    def gen_min_rule(model, g):
        return model.pg[g] >= model.gen_min[g]
    model.c_gen_min = Constraint(model.GEN, rule=gen_min_rule)

    # Branch power flow equation
    def br_flow_rule(model, br):
        return model.pf[br] == sum(model.Bf[br, b]*model.va[b] 
                                    for b in model.BUS)
    model.c_br_flow = Constraint(model.BRCH, rule=br_flow_rule)

    # Branch power flow limits
    def br_max_rule(model, br):
        return model.pf[br] <= model.br_max[br]
    model.c_br_max = Constraint(model.BRCH, rule=br_max_rule)

    def br_min_rule(model, br):
        return model.pf[br] >= model.br_min[br]
    model.c_br_min = Constraint(model.BRCH, rule=br_min_rule)

    # DC power flow equation
    def pf_rule(model, b):
        return sum(model.gen_map[b, g]*model.pg[g] for g in model.GEN) \
            - sum(model.load_map[b, l]*model.load[l] for l in model.LOAD)\
            == sum(model.B[b, b_]*model.va[b_] for b_ in model.BUS)
    model.c_pf = Constraint(model.BUS, rule=pf_rule)

    # DC line power balance
    def dcline_rule(model, dc):
        return model.pg[model.dc_idx_f[dc]] + model.pg[model.dc_idx_t[dc]] == 0
    model.c_dcline = Constraint(model.DC, rule=dcline_rule)

    # Interface flow limit
    def if_max_rule(model, i):
        return sum(model.id_br_dir[i][br]*model.pf[model.if_br_idx[i][br]] 
                    for br in range(len(model.if_br_idx))) <= model.if_lim_max[i]
    model.c_if_lim_max = Constraint(model.IFACE, rule=if_max_rule)

    def if_min_rule(model, i):
        return sum(model.id_br_dir[i][br]*model.pf[model.if_br_idx[i][br]] 
                    for br in range(len(model.if_br_idx))) >= model.if_lim_min[i]
    model.c_if_lim_min = Constraint(model.IFACE, rule=if_min_rule)

    # Define objective
    def cost_rule(model):
        cost = sum(model.gencost_0[g] + model.gencost_1[g]*model.pg[g]
                    for g in model.GEN)
        return cost
    model.obj = Objective(rule=cost_rule, sense=minimize,
                            doc="Minimize total generation cost")

    return model    

def multi_opf(B, Bf, gen_map, load_map, load, gencost_0, gencost_1, dc_idx_f, dc_idx_t,
                gen_min, gen_max, br_min, br_max, ramp_down, ramp_up, if_map, if_lims):
    '''
    Multi-period OPF problem.

    Parameters:
        A tuple of OPF network parameters and constraints.
    
    Returns:
        model (pyomo.core.base.PyomoModel.ConcreteModel): Pyomo model for multi-period OPF problem.
    '''

    # Check dimensions of the input data
    if (gen_min.shape != gen_max.shape) or (ramp_down.shape != ramp_up.shape):
        raise ValueError('Found mismatch in generator constraint dimensions!')
        
    if (br_min.shape != br_max.shape):
        raise ValueError('Found mismatch in branch flow limit array dimensions!')
    
    # Get problem dimensions
    NB, NG = gen_map.shape
    NT, NL = load.shape
    NBR = br_min.shape[0]

    model = ConcreteModel(name='multi-period OPF')

    # Define variables
    model.PG = Var(range(NT), range(NG), within=Reals, initialize=1)  # Real power generation
    model.Va = Var(range(NT), range(NB), within=Reals, initialize=0,
                   bounds=(-2*np.pi, 2*np.pi))  # Bus phase angle
    model.PF = Var(range(NT), range(NBR), within=Reals, initialize=0) # Branch power flow
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

    for t in range(NT):
        # Line flow limit constraints
        for br in range(NBR):
            model.c_br_flow.add(model.PF[t, br] == sum(Bf[br, b]*model.Va[t, b] for b in range(NB)))
            model.c_br_max.add(model.PF[t, br] <= br_max[br])
            model.c_br_min.add(model.PF[t, br] >= br_min[br])

        # Generation capacity limit
        for g in range(NG):
            model.c_gen_max.add(model.PG[t, g] <= gen_max[g])
            model.c_gen_min.add(model.PG[t, g] >= gen_min[g])

        # DC Power flow constraint
        for b in range(NB):
            model.c_pf.add(sum(gen_map[b, g]*model.PG[t, g] for g in range(NG)) - sum(load_map[b, l]*load[t, l] for l in range(NL))
                            == sum(B[b, b_]*model.Va[t, b_] for b_ in range(NB)))
        
        # DC line power balance constraint
        for idx_f, idx_t in zip(dc_idx_f, dc_idx_t):
            model.c_dcline.add(model.PG[t, idx_f] == -model.PG[t, idx_t])

         # Interface flow limits
        for n in range(len(if_lims)):
            if_id, if_lims_min, if_lims_max = if_lims[n, :]
            br_dir_idx = if_map[(if_map[:,0] == int(if_id)), 1]
            br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
            model.c_if_min.add(if_lims_min <= sum(br_dir[i]*model.PF[t, br_idx[i]] for i in range(len(br_idx))))
            model.c_if_max.add(if_lims_max >= sum(br_dir[i]*model.PF[t, br_idx[i]] for i in range(len(br_idx))))

    for t in range(NT-1):
        # Ramp rate limit
        for g in range(NG):
            model.c_gen_ramp_down.add(-model.PG[t+1, g] + model.PG[t, g] <= ramp_down[g])
            model.c_gen_ramp_up.add(model.PG[t+1, g] - model.PG[t, g] <= ramp_up[g])

    def cost(model, gencost_0, gencost_1):
        cost = 0
        for t in range(NT):
            cost += sum(gencost_0[g] for g in range(NG)) \
                + sum(gencost_1[g]*model.PG[t, g] for g in range(NG))

        return cost

    model.obj = Objective(expr=cost(model, gencost_0, gencost_1), sense=minimize)

    return model

def check_model_status(results):
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

def process_ppc(ppc, load):
    '''
    Process PyPower case to get constraint value for the OPF problem.

    Parameters:
        ppc (dict): PyPower case in python dictionary.
        load (numpy.ndarray): A 2-d array of load at each timestep at each bus
    
    Returns:
        Parameters of the network and constraints.
    '''

    # Remove user functions
    del ppc['userfcn']

    # Format genfuel and bus_name strings
    ppc['genfuel'] = np.array([str(x[0][0]) for x in ppc['genfuel']])
    ppc['bus_name'] = np.array([str(x[0][0]) for x in ppc['bus_name']])

    # Format interface limit data
    ppc['if'] = {
        'map': ppc['if'][0][0][0],
        'lims': ppc['if'][0][0][1]
    }
    ppc = pp.toggle_iflims(ppc, 'on')

    # Convert baseMVA to float
    ppc['baseMVA'] = float(ppc['baseMVA'])

    # Convert DC line to generators and add to gen matrix
    ppc_dc, num_dcline = dcline2gen(ppc)

    # Convert to internal indexing
    ppc_int = pp.ext2int(ppc_dc)

    baseMVA, bus, gen, branch, gencost = \
        ppc_int["baseMVA"], ppc_int["bus"], ppc_int["gen"], ppc_int["branch"], ppc_int["gencost"]

    # Generator info
    gen_bus = gen[:, GEN_BUS].astype(int)  # what buses are they at?

    # build B matrices and phase shift injections
    B, Bf, _, _ = pp.makeBdc(baseMVA, bus, branch)
    B = B.todense()
    Bf = Bf.todense()

    # Problem dimensions
    NG = gen.shape[0]  # Number of generators
    NB = bus.shape[0]  # Number of buses
    NBR = branch.shape[0]  # Number of lines
    NL = np.count_nonzero(bus[:, PD])  # Number of loads

    # Get mapping from generator to bus
    gen_map = np.zeros((NB, NG))
    gen_map[gen_bus, range(NG)] = 1

    # Get index of DC line converted generators in internal indexing
    gen_i2e = ppc_int['order']['gen']['i2e']
    dc_idx_f = gen_i2e[NG-num_dcline*2: NG-num_dcline]
    dc_idx_t = gen_i2e[NG-num_dcline: NG]

    # Get mapping from load to bus
    load_map = np.zeros((NB, NL))
    load_bus = np.nonzero(bus[:, PD])[0]
    for i in range(len(load_bus)):
        load_map[load_bus[i], i] = 1

    # Generator capacity limit in p.u.
    gen_max = gen[:, PMAX]/baseMVA
    gen_min = gen[:, PMIN]/baseMVA

    # Generator upward and downward ramp limit in p.u.
    ramp_up = gen[:, RAMP_30]*2/baseMVA
    ramp_down = np.min([gen_max, ramp_up], axis=0)

    # Line flow limit in p.u.
    br_max = branch[:, RATE_A]/baseMVA
    # Replace default value 0 to 999
    br_max[br_max == 0] = 999.99
    br_min = - br_max

    # Linear cost coefficients in p.u.
    gencost_1 = gencost[:, COST]*baseMVA
    gencost_0 = gencost[:, COST+1]

    # Get interface limit information
    if_map = ppc_int['if']['map']
    if_lims = ppc_int['if']['lims']
    if_lims[:,1:] = if_lims[:,1:]/baseMVA
    br_dir, br_idx = np.sign(if_map[:,1]), np.abs(if_map[:,1]).astype(int)
    if_map[:, 1] = br_dir*(br_idx-1)
    
    # Convert load to p.u.
    load = load/baseMVA

    return (ppc_int, B, Bf, gen_map, load_map, load, gencost_0, gencost_1, dc_idx_f, dc_idx_t,
            gen_min, gen_max, br_min, br_max, ramp_down, ramp_up, if_map, if_lims)

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

def opf_results_single(model_single, ppc_int):
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
    baseMVA = ppc_int['baseMVA']
    num_bus = ppc_int['bus'].shape[0]

    # Power generation
    results_pg = np.array(model_single.PG[:]())*baseMVA
    gen_order = ppc_int['order']['gen']['e2i']
    results_pg = pd.Series(results_pg, index=gen_order).sort_index()
    
    # Bus phase angle
    results_va = np.array(model_single.Va[:]())*180/np.pi
    # Just to compare with PyPower
    results_va = results_va - 73.4282
    # Convert negative numbers to 0-360
    results_va = np.where(results_va < 0, results_va+360, results_va)
    results_va = pd.Series(results_va)

    # Branch power flow
    branch_pf = np.array(model_single.PF[:]())*baseMVA
    results_pf = pd.Series(branch_pf)

    # Interface flow
    if_lims = ppc_int['if']['lims']
    if_map = ppc_int['if']['map']
    if_flow = np.zeros(len(if_lims))
    for n in range(len(if_lims)):
        if_id = if_lims[n, 0]
        br_dir_idx = if_map[(if_map[:,0] == int(if_id)), 1]
        br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
        if_flow[n] = sum(br_dir[i]*branch_pf[br_idx[i]-1] for i in range(len(br_idx)))
        if_flow[n] = sum(br_dir[i]*branch_pf[br_idx[i]-1] for i in range(len(br_idx)))
    results_if = pd.Series(if_flow)

    bus_lmp = np.zeros(num_bus)
    for i in range(num_bus):
        bus_lmp[i] = np.abs(model_single.dual[model_single.c_pf[i+1]])/baseMVA
    results_lmp = pd.Series(bus_lmp)

    results = {
        'PG': results_pg,
        'VA': results_va,
        'PF': results_pf,
        'IF': results_if,
        'LMP': results_lmp
    }

    return results


def opf_results_multi(model_multi, ppc_int, start_time, end_time):
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
    baseMVA = ppc_int['baseMVA']
    timestamp_list = pd.date_range(start_time, end_time, freq='1H')
    num_time = timestamp_list.shape[0]
    num_gen = ppc_int['gen'].shape[0]
    num_bus = ppc_int['bus'].shape[0]
    num_branch = ppc_int['branch'].shape[0]

    # Power generation
    results_pg = np.array(model_multi.PG[:,:]()).reshape(num_time, num_gen)*baseMVA
    gen_order = ppc_int['order']['gen']['e2i']
    results_pg = pd.DataFrame(results_pg, index=timestamp_list,
                                columns=gen_order).sort_index(axis=1)
    
    # Bus phase angle
    results_va = np.array(model_multi.Va[:,:]()).reshape(num_time, num_bus)*180/np.pi
    # Just to compare with PyPower
    results_va = results_va - 73.4282
    # Convert negative numbers to 0-360
    results_va = np.where(results_va < 0, results_va+360, results_va)
    results_va = pd.DataFrame(results_va, index=timestamp_list)

    # Branch power flow
    branch_pf = np.array(model_multi.PF[:,:]()).reshape(num_time, num_branch)*baseMVA
    results_pf = pd.DataFrame(branch_pf, index=timestamp_list)

    # Interface flow
    if_lims = ppc_int['if']['lims']
    if_map = ppc_int['if']['map']
    if_flow = np.zeros((num_time, len(if_lims)))
    for t in range(num_time):
        for n in range(len(if_lims)):
            if_id = if_lims[n, 0]
            br_dir_idx = if_map[(if_map[:,0] == int(if_id)), 1]
            br_dir, br_idx = np.sign(br_dir_idx), np.abs(br_dir_idx).astype(int)
            if_flow[t, n] = sum(br_dir[i]*branch_pf[t, br_idx[i]-1] for i in range(len(br_idx)))
            if_flow[t, n] = sum(br_dir[i]*branch_pf[t, br_idx[i]-1] for i in range(len(br_idx)))
    results_if = pd.DataFrame(if_flow, index=timestamp_list)

    bus_lmp = np.zeros(num_time*num_bus)
    for i in range(num_time*num_bus):
        bus_lmp[i] = np.abs(model_multi.dual[model_multi.c_pf[i+1]])/baseMVA
    results_lmp = bus_lmp.reshape(num_time, num_bus)
    results_lmp = pd.DataFrame(results_lmp, index=timestamp_list)

    results = {
        'PG': results_pg,
        'VA': results_va,
        'PF': results_pf,
        'IF': results_if,
        'LMP': results_lmp
    }

    return results
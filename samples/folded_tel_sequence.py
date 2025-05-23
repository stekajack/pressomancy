from pressomancy.simulation import Simulation, TelSeq, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging

N_avog = 6.02214076e23
sigma = 1.
rho_si = 0.6*N_avog
no_obj=6
N = int(no_obj/3)
vol = N/rho_si
box_l = pow(vol, 1/3)
_box_l = box_l/0.4e-09
box_dim = _box_l*np.ones(3)

sheets_per_quad = 3
part_per_filament = 2

sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys()
logging.info(f'box_dim: {sim_inst.sys.box_l}')
bond_hndl=BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=1.5))
quartet_config = Quartet.config.specify(bond_handle=bond_hndl,type='broken', espresso_handle=sim_inst.sys)
quartets = [Quartet(config=quartet_config) for x in range(no_obj)]

grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=sim_inst.sys, bonding_mode='ftf',bond_handle=bond_quad, size=np.sqrt(3)*5) for elem in grouped_quartets]
quadriplex = [Quadriplex(config=elem) for elem in quadriplex_config_list]

diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=np.sqrt(2)*(2*2.), d_r_max=2*1.5))
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
tel_config_list = [TelSeq.config.specify(n_parts=part_per_filament, espresso_handle=sim_inst.sys, associated_objects=elem, size=quadriplex[0].params['size']*part_per_filament+np.sqrt(3)*bond_quad.r_0+(part_per_filament-1),bond_handle=bond_quad,diag_bond_handle=diag_bond,spacing=6.) for elem in grouped_quadriplexes]
telomeres = [TelSeq(config=elem) for elem in tel_config_list]
sim_inst.store_objects(telomeres)
sim_inst.set_objects(telomeres)

for telomere in telomeres:
    telomere.wrap_into_Tel()

sim_inst.set_steric_custom(
    pairs=[('real', 'circ'), ('real', 'squareA'), ('real', 'squareB'), ('virt', 'circ'), ('virt', 'squareA'), ('virt', 'squareB')], wca_eps=[1, 1, 1, 1, 1, 1], sigma=[1, 1, 1, 1, 1, 1])

sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)
sim_inst.sys.integrator.run(0)
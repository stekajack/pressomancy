from pressomancy.simulation import Simulation, Filament, Quartet, Quadriplex
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
bond_hndl=espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=1.5)
sim_inst.sys.bonded_inter.add(bond_hndl)
quartets = [Quartet(sigma=sigma, n_parts=25, type='broken', bond_handle=bond_hndl, espresso_handle=sim_inst.sys) for x in range(no_obj)]
sim_inst.store_objects(quartets)

grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
bond_quad = espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5)
sim_inst.sys.bonded_inter.add(bond_quad)
quadriplex = [Quadriplex(sigma=np.sqrt(Quartet.n_parts)*Quartet.sigma, quartet_grp=elem, espresso_handle=sim_inst.sys, bonding_mode='ftf',bond_handle=bond_quad, size=6.) for elem in grouped_quartets]
sim_inst.store_objects(quadriplex)
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
filaments = [Filament(sigma=6, n_parts=part_per_filament,
                        espresso_handle=sim_inst.sys, associated_objects=elem, size=6*part_per_filament) for elem in grouped_quadriplexes]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)
diag_bond = espressomd.interactions.FeneBond(k=10., r_0=np.sqrt(2)*(2*2.), d_r_max=2*1.5)

sim_inst.sys.bonded_inter.add(diag_bond)

for filament in filaments:
    filament.wrap_into_Tel(bond_handles=[bond_quad, diag_bond])

sim_inst.set_steric_custom(
    pairs=[('real', 'circ'), ('real', 'squareA'), ('real', 'squareB'), ('virt', 'circ'), ('virt', 'squareA'), ('virt', 'squareB')], wca_eps=[1, 1, 1, 1, 1, 1], sigma=[1, 1, 1, 1, 1, 1])


sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)
sim_inst.sys.integrator.run(0)

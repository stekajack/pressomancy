from pressomancy.simulation import Simulation, Crowder, Filament, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging
N_avog = 6.02214076e23

sigma = 1.
rho_si = 0.6*N_avog
no_obj=30
N = int(no_obj/3)
vol = N/rho_si
box_l = pow(vol, 1/3)
_box_l = box_l/0.4e-09
box_dim = _box_l*np.ones(3)
_rho = N/pow(_box_l, 3)

sheets_per_quad = 3
part_per_filament = 2
no_crowders=10
part_per_ligand=2

sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys()
logging.info(f'box_dim: {sim_inst.sys.box_l}')

quartet_configuration = Quartet.config.specify(size=2., n_parts=25, espresso_handle=sim_inst.sys,type='solid')
quartets = [Quartet(config=quartet_configuration) for x in range(no_obj)]
sim_inst.store_objects(quartets)

bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
quadriplex_configuration_list = [Quadriplex.config.specify(size=6., espresso_handle=sim_inst.sys, bond_handle=bond_quad, associated_objects=elem) for elem in grouped_quartets]

quadriplex = [Quadriplex(config=configuration) for configuration in quadriplex_configuration_list]
sim_inst.store_objects(quadriplex)

bond_pass = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
filament_configuration_list = [Filament.config.specify(sigma=6,size=6*part_per_filament, n_parts=part_per_filament, espresso_handle=sim_inst.sys, bond_handle=bond_pass, associated_objects=elem) for elem in grouped_quadriplexes]
filaments = [Filament(config=configuration) for configuration in filament_configuration_list]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)


for filament in filaments:        
    filament.bond_quadriplexes()

sim_inst.sys.integrator.run(0)
crowder_configuration=Crowder.config.specify(sigma=6., size=6., espresso_handle=sim_inst.sys)
crowders = [Crowder(config=crowder_configuration)
            for x in range(no_crowders)]
sim_inst.store_objects(crowders)
grouped_crowders = [crowders[i:i+part_per_ligand]
                for i in range(0, len(crowders), part_per_ligand)]

bender_pass = BondWrapper(espressomd.interactions.FeneBond(
    k=10, r_0=6, d_r_max=6*1.5))
filament_configuration_list = [Filament.config.specify(sigma=6,size=6*part_per_ligand, n_parts=part_per_ligand, espresso_handle=sim_inst.sys, bond_handle=bender_pass, associated_objects=elem) for elem in grouped_crowders]

filaments = [Filament(config=elem) for elem in filament_configuration_list]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)

for filament in filaments:
    filament.bond_center_to_center(type_key='crowder')
    

sim_inst.set_steric(key=('real', 'virt','crowder'), wca_eps=1.)


for el in quadriplex:
    el.add_patches_triples()
sim_inst.set_vdW(key=('patch',), lj_eps=5, lj_size=2.)

sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)
sim_inst.sys.integrator.run(0)


from pressomancy.simulation import Simulation, Crowder, Filament, Quartet, Quadriplex
import espressomd
import os
import sys as sysos
import numpy as np
N_avog = 6.02214076e23
import gc

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
print('box_dim: ', sim_inst.sys.box_l)
print(Quadriplex.numInstances)
quadriplex_instances = [obj for obj in gc.get_objects() if isinstance(obj, Quadriplex)]
for idx, instance in enumerate(quadriplex_instances, start=1):
    print(f"Instance {idx}: {instance}")
quartets = [Quartet(sigma=sigma, n_parts=25, type='solid', espresso_handle=sim_inst.sys, fene_k=10, fene_r0=1) for x in range(no_obj)]
sim_inst.store_objects(quartets)

grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]

quadriplex = [Quadriplex(sigma=np.sqrt(Quartet.n_parts)*Quartet.sigma, quartet_grp=elem,
                         espresso_handle=sim_inst.sys, fene_k=10, fene_r0=2., bending_k=0, bending_angle=np.pi, bonding_mode='ftf', size=6.) for elem in grouped_quartets]
sim_inst.store_objects(quadriplex)
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
filaments = [Filament(sigma=6, n_parts=part_per_filament,
                        espresso_handle=sim_inst.sys, associated_objects=elem, size=6*part_per_filament) for elem in grouped_quadriplexes]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)
bond_pass = espressomd.interactions.FeneBond(
    k=Quadriplex.fene_k, r_0=Quadriplex.fene_r0, d_r_max=Quadriplex.fene_r0*1.5)
for filament in filaments:        
    filament.bond_quadriplexes(bond_handle=bond_pass)

sim_inst.sys.integrator.run(0)
crowders = [Crowder(sigma=6., espresso_handle=sim_inst.sys)
            for x in range(no_crowders)]
sim_inst.store_objects(crowders)
grouped_crowders = [crowders[i:i+part_per_ligand]
                for i in range(0, len(crowders), part_per_ligand)]
filaments = [Filament(sigma=6, n_parts=part_per_ligand,
                    espresso_handle=sim_inst.sys, associated_objects=elem, size=6*part_per_ligand) for elem in grouped_crowders]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)

bender_pass = espressomd.interactions.FeneBond(
    k=10, r_0=6, d_r_max=6*1.5)
for filament in filaments:
    filament.bond_center_to_center(bond_handle=bender_pass,type_key='crowder')
    

sim_inst.set_steric(key=('real', 'virt','crowder'), wca_eps=1.)


for el in quadriplex:
    el.add_patches_triples()
sim_inst.set_vdW(key=('patch',), lj_eps=5, lj_size=2.)

for el in quartets:
    el.exclude_self_interactions()
sim_inst.sys.integrator.run(0)

sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)
sim_inst.sys.integrator.run(0)


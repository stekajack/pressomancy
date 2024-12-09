import espressomd
from pressomancy.simulation import Simulation, Filament
import numpy as np


sigma = 1.
n_part_tot = 10
density=0.001
box_dim = np.power((4*n_part_tot*np.pi*np.power(sigma/2, 3))/(3*density), 1/3)*np.ones(3)
print('box_dim: ', box_dim)
sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys()

filaments = [Filament(sigma=sigma, n_parts=2, size=2, espresso_handle=sim_inst.sys) for x in range(5)]
bond_hndl=espressomd.interactions.FeneBond(k=10, d_r_max=3*sigma, r_0=0)
sim_inst.sys.bonded_inter.add(bond_hndl)

sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)
for filament in filaments:
    filament.add_anchors(type_key='real')
    filament.bond_overlapping_virtualz(bond_hndl)
    filament.add_dipole_to_embedded_virt(type_name='real',dip_magnitude=1.)

sim_inst.set_vdW(key=('real',),lj_eps=3.)

sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)
sim_inst.set_H_ext()
sim_inst.set_H_ext(H=(0,0,6.66))
sim_inst.get_H_ext()

# part_list, dip_magnitude, H_ext
H_ext=sim_inst.get_H_ext()
pats_to_magnetize=sim_inst.sys.part.select(lambda p:p.type==sim_inst.part_types['to_be_magnetized'])
sim_inst.sys.integrator.run(0)
sim_inst.magnetize(pats_to_magnetize,1.732,H_ext=H_ext)
sim_inst.sys.integrator.run(1)
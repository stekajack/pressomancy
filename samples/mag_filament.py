import espressomd
from pressomancy.simulation import Simulation, Filament
import numpy as np


sigma = 1.
n_part_tot = 1000
density=0.001
box_dim = np.power((4*n_part_tot*np.pi*np.power(sigma/2, 3))/(3*density), 1/3)*np.ones(3)
print('box_dim: ', box_dim)
sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys()

filaments = [Filament(sigma=sigma, n_parts=10, size=10, espresso_handle=sim_inst.sys) for x in range(100)]
bond_hndl=espressomd.interactions.FeneBond(k=10, d_r_max=3*sigma, r_0=0)
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)
for filament in filaments:
    filament.add_anchors(type_key='real')
    filament.bond_overlapping_virtualz(bond_hndl)
    filament.add_dipole_to_embedded_virt(type_name='real',dip_magnitude=1.)

sim_inst.set_vdW(key=('real',),lj_eps=3.)

sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)
sim_inst.sys.integrator.run(0)
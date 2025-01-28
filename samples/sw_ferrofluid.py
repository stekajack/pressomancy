import numpy as np
from pressomancy.simulation import Simulation, SWPart

n_part=100
vol_frac=0.01
xi=5
kT_KVm_inv=5
tstp=0.001
LJ_SIGMA = 1
LJ_EPSILON = 1
LJ_CUT = 2**(1. / 6.) * LJ_SIGMA

temperature = 1
t_=3.44e-08
H_reduced=2.8569745177717567
HK_inv=0.175
h=0.5
dipole_moment=1.75
gamma_T=74.87
gamma_R=24.96
box_l=17.36465693*np.ones(3)
tau0_inv=735412234.8230474

sim_inst = Simulation(box_dim=box_l)
sim_inst.set_sys()
configuration=SWPart.config.specify(sigma=LJ_SIGMA, size=LJ_SIGMA, kT_KVm_inv=kT_KVm_inv, dipm=dipole_moment, dt_incr=t_*tstp, tau0_inv=tau0_inv, HK_inv=HK_inv, espresso_handle=sim_inst.sys)

sw_parts=[SWPart(config=configuration) for _ in range(n_part)]
sim_inst.store_objects(sw_parts)
sim_inst.set_objects(sw_parts)

sim_inst.set_steric(key=('sw_real', ), wca_eps=LJ_EPSILON)
sim_inst.set_H_ext(H=(0, 0, H_reduced))

sim_inst.sys.thermostat.set_langevin(kT=temperature, gamma=gamma_T,
                               gamma_rotation=gamma_R, seed=sim_inst.seed)

sim_inst.sys.integrator.run(0)

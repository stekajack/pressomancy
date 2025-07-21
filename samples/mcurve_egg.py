import numpy as np
from pressomancy.simulation import Simulation, EGGPart


xi = 1
sigma = 1
LJ_EPSILON=1.
n_steps = 100
dipm=1.
vol_frac = 0.001
n_part = 100

dt = 0.004

box_l = (np.pi/6. * n_part / vol_frac)**(1./3) 
sim_inst = Simulation(box_dim=box_l*np.ones(3))
sim_inst.set_sys(timestep=dt,have_quaternion=True)

egg_parts=[EGGPart(config=EGGPart.config.specify(espresso_handle=sim_inst.sys)) for _ in range(n_part)]
sim_inst.store_objects(egg_parts)
sim_inst.set_objects(egg_parts)
sim_inst.sys.thermostat.turn_off()
sim_inst.sys.thermostat.set_brownian(kT=1., gamma=1.0, seed=sim_inst.seed, act_on_virtual = True)
sim_inst.sys.integrator.set_brownian_dynamics()

sim_inst.set_steric(key=('real', ), wca_eps=LJ_EPSILON)
sim_inst.set_H_ext(H=(0, 0, xi))

sim_inst.sys.integrator.run(n_steps)


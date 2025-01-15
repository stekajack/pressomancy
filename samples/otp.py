
import numpy as np
import espressomd
import espressomd.interactions
import logging
from pressomancy.simulation import Simulation, OTP

lj_eps = 1.8350838581351132
lj_sig = 0.483
lj_cut = 2.5 * lj_sig

kT=1
n_part =10*3
density=2.775

box_l=pow(n_part/density,1/3.)
logging.info('density: ', density)
logging.info('box_l: ', box_l)
logging.info('vol_fract [%]: ', (n_part*np.pi*pow(lj_sig,3)*pow(6,-1)/pow(box_l,3))*100)

sim_inst = Simulation(box_dim=box_l*np.ones(3))
sim_inst.set_sys(timestep=0.001)

rig_long = espressomd.interactions.RigidBond(r=0.7191944807239401, ptol=1e-12, vtol=1e-12) 
rig_short = espressomd.interactions.RigidBond(r=0.483, ptol=1e-12, vtol=1e-12)
sim_inst.sys.bonded_inter.add(rig_long)
sim_inst.sys.bonded_inter.add(rig_short)
opts = [OTP(sigma=lj_sig,long_side=(rig_long,0.7191944807239401), short_side=(rig_short,0.483), espresso_handle=sim_inst.sys) for _ in range(10)]
sim_inst.store_objects(opts)
sim_inst.set_objects(opts)

sim_inst.set_vdW(key=('otp',), lj_eps=lj_eps, lj_size=lj_sig)
sim_inst.sys.integrator.set_vv()
sim_inst.sys.thermostat.set_langevin(kT=kT, gamma=1.0, seed=sim_inst.seed)

sim_inst.sys.integrator.run(0)



import espressomd
import espressomd.magnetostatics

from pressomancy.simulation import Simulation, EspressoPart

import numpy as np

espressomd.assert_features(['WCA', 'DIPOLES', 'DP3M',
                            'VIRTUAL_SITES', 'VIRTUAL_SITES_RELATIVE',
                            'EXTERNAL_FORCES'])

np.random.seed(42)

# System params
N_HM = 100
SIZE_HM = 1.
r_HM= SIZE_HM/2
BOX_SIZE = np.cbrt(4) * ( np.cbrt( 100 * 4/3*np.pi / 0.3 ) * r_HM )

box_HM = [BOX_SIZE, BOX_SIZE, BOX_SIZE/4]

# INITIALIZE SYSTEM
system = Simulation(box_dim=[BOX_SIZE]*3)
system.set_sys(time_step=0.001)
system.seed = 42

# Add HM particles
config_HM = EspressoPart.config.specify(type_name='HM', type=3, dipm=2., rotation=[True, True, True], espresso_handle=system.sys, sigma=SIZE_HM, size=SIZE_HM)

part_HM = [EspressoPart(config=config_HM) for _ in range(N_HM)]
system.store_objects(part_HM)
box_HM_pos = box_HM[:2] + [box_HM[2] - 2*r_HM]
system.set_objects(part_HM, box_HM_pos, shift=[0,0,r_HM])

system.set_steric(key=('HM',), wca_eps=1., sigma=config_HM['sigma'])

# Add substrate particles
config_SUBSTRATE = EspressoPart.config.specify(type_name='SUBSTRATE', type=4, virtual=True, fix=[True, True, True], espresso_handle=system.sys, size=1.)

n_substrate_x = int(np.ceil(system.sys.box_l[0]))
n_substrate_y = int(np.ceil(system.sys.box_l[1]))
N_SUBSTRATE= n_substrate_x * n_substrate_y
pos_x, pos_y = np.meshgrid( np.linspace(0.5, system.sys.box_l[0]-0.5, n_substrate_x),
                            np.linspace(0.5, system.sys.box_l[1]-0.5, n_substrate_y) )
pos = np.column_stack((pos_x.ravel(), pos_y.ravel(), (np.zeros(N_SUBSTRATE) - 0.5) ))
part_SUBSTRATE = [EspressoPart(config=config_SUBSTRATE) for _ in range(N_SUBSTRATE)]
system.store_objects(part_SUBSTRATE)
system.place_objects(part_SUBSTRATE, pos)

system.set_steric_custom( (('HM', 'SUBSTRATE'),), wca_eps=(1E6,),
                         sigma=( (config_SUBSTRATE['size'] + config_HM['size'])/2 / 2**(1/6), ) )

# START THERMALIZATION

# Add temporary box particles
system.add_box_constraints(0, sides=['no-sides'], top=box_HM[2],
                            inter='wca', types_=EspressoPart.part_types['HM'])

# First relaxation (low T)
system.sys.thermostat.set_langevin(kT=1E-3, gamma=100, seed=42)
system.sys.integrator.run(1)

assert system.sys.analysis.min_dist(p1=[EspressoPart.part_types['HM']], p2=[EspressoPart.part_types['HM']]) >= 2*r_HM
assert system.sys.analysis.min_dist(p1=[EspressoPart.part_types['SUBSTRATE']], p2=[EspressoPart.part_types['HM']]) >= (0.5+r_HM)

# Second relaxation (high T)
system.sys.thermostat.set_langevin(kT=0.5, gamma=10, seed=42)
system.sys.time_step = 0.0001
system.sys.integrator.run(1)

# Remove temporary box particles
system.remove_box_constraints()

# Stuck the bottom layer particles to the z=r_HM plane
#  (restrict movement in z direction)
z_tmp = system.sys.part.select(type=EspressoPart.part_types['SUBSTRATE']).pos[0,2] + 0.5 + r_HM
for particle_HM in system.sys.part.select(type=EspressoPart.part_types['HM']):
    if particle_HM.pos[2] < (z_tmp + r_HM/4): # chose at which heights to capture HMs
        particle_HM.pos = [particle_HM.pos[0], particle_HM.pos[1], z_tmp]
        particle_HM.fix = [False, False, True]

# Add random elastic bonds between HM
system.random_harmonic_bonds(r_catch=5., bond_k=(0.001, 0.01), max_bonds=6, object_types=(EspressoPart,), part_types=(('HM','HM'),))

# Third relaxation (with elastic bonds)
system.sys.thermostat.set_langevin(kT=1E-3, gamma=10, seed=42)
system.sys.integrator.run(0)

# SIMULATE MAE
system.sys.time = 0.
system.sys.time_step = 0.001

# Add thermostat
system.sys.thermostat.set_langevin(kT=1E-3, gamma=10, seed=42)

# Add magnetic dipole interactions - p3m with Dioplar Layer Correction (DLC)
p3m = espressomd.magnetostatics.DipolarP3M(prefactor=1.,mesh=32, accuracy=1E-4)
mdlc_gap = r_HM + 1
mdlc = espressomd.magnetostatics.DLC(actor=p3m, maxPWerror=1E-5, gap_size=mdlc_gap)
system.sys.actors.add(mdlc)

# Add non-interating walls to stop simulation when particles enter the DLC gap region
top_wall = system.add_box_constraints(0, sides=['top','bottom'], top=(system.sys.box_l[2] - mdlc_gap))

# Gradually increasing dipole moments
dipm= 0.
for increase in range(10):
    dipm += 0.2
    system.sys.part.select(type=EspressoPart.part_types['HM']).dipm = dipm
    system.sys.integrator.run(0)

# Gradually increasing magnetic field
ext_H_z = 0
for increase in range(10):
    ext_H_z += 0.7
    system.set_H_ext(H=(0,0,ext_H_z))
    system.sys.integrator.run(0)

assert system.get_H_ext()[2] == ext_H_z and np.isclose(ext_H_z, 7)
assert system.get_H_ext()[0] == system.get_H_ext()[1] ==0

# Final relaxation with full external field applied
system.sys.time_step = 0.0005
system.sys.integrator.run(1)

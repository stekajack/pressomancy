import espressomd
import espressomd.magnetostatics
import espressomd.checkpointing

espressomd.assert_features(['WCA', 'ROTATION', 'DIPOLES', 'DP3M',
                            'VIRTUAL_SITES', 'VIRTUAL_SITES_RELATIVE',
                            'EXTERNAL_FORCES'])

from pressomancy.simulation import Simulation, Elastomer, PointDipolePermanent, PointDipoleSuperpara

import numpy as np

#################

# Simulation parameters
sim_params = {'DENS_PART': 0.3, 'BOND_K': "soft", 'HEIGHT': 6,
              'N_FULL_BOX': 120*4, 'seed': 1,
              'H': 1}

# System params
DENS_PART = sim_params['DENS_PART']
MAE_LAYER_HEIGHT_parts = sim_params['HEIGHT'] # in part size units
N_M_FULL_BOX = sim_params['N_FULL_BOX']

SIGMA_PART = 1.
SIZE_PART = SIGMA_PART
R_PART= SIZE_PART/2

BONDS_MAX_LENGHT_A = 5.
BOND_K_A = sim_params['BOND_K']
valid_bond_limits = {'soft': (0.001, 0.01), 'hard': (0.01, 0.1)}
BOND_LIMITS_A = valid_bond_limits.get(BOND_K_A)
if BOND_LIMITS_A is None:
   raise ValueError(f"Invalid bond type: {BOND_K_A}. Valid --K values -> {valid_bond_limits.keys()}")

MAE_LAYER_HEIGHT = MAE_LAYER_HEIGHT_parts * SIZE_PART # in system units
BOX_SIZE = np.cbrt( N_M_FULL_BOX * 4/3*np.pi / 0.3 ) * R_PART
BOX_Z_MAX = 4 * BOX_SIZE

N_PART = round(DENS_PART * BOX_SIZE**2 * MAE_LAYER_HEIGHT / ( 4/3 * np.pi * R_PART**3))

assert MAE_LAYER_HEIGHT<=BOX_Z_MAX
assert DENS_PART<=0.3

box_l = [BOX_SIZE, BOX_SIZE, BOX_Z_MAX]
box_E = [BOX_SIZE, BOX_SIZE, MAE_LAYER_HEIGHT]

dens_A = N_PART * 4/3 * np.pi * R_PART**3 / np.prod(box_E)

assert dens_A - DENS_PART < 0.001, f"DENS_{dens_A}"

# INITIALIZE SYSTEM
system = Simulation(box_dim=box_l)
system.seed = sim_params['seed']
system.set_sys(time_step=0.001)

config_pdp = PointDipolePermanent.config.specify(dipm=1., espresso_handle=system.sys)
config_pds = PointDipoleSuperpara.config.specify(dipm=1., espresso_handle=system.sys)
n_pdp = int(N_PART/2); n_pds = N_PART - n_pdp
associated_objects = [PointDipolePermanent(config=config_pdp) for _ in range(n_pdp)] + [PointDipoleSuperpara(config=config_pds) for _ in range(n_pds)]
assert len(associated_objects) == N_PART
config_E = Elastomer.config.specify(box_E=box_E, box_E_shift=[0,0,0.5], n_parts=N_PART, associated_objects=associated_objects, bond_K_lims=BOND_LIMITS_A, size=SIZE_PART, sigma=SIGMA_PART, espresso_handle=system.sys, seed=system.seed)
elastomer=[Elastomer(config=config_E) for _ in range(1)]
system.store_objects(elastomer)
system.set_objects(elastomer, box_lengths=elastomer[0].params['box_E'], shift=elastomer[0].params['box_E_shift'])
elastomer= elastomer[0]

print(set(system.sys.part.all().type))

system.set_steric(key=("pdp_real", "pds_real"))

# must add non_bonded interactions before creating substrate
elastomer.create_substrate(geometry='part')

elastomer.mix_elastomer_stuff(test=True)

elastomer.cure_elastomer(test=True)

elastomer.relax_langevin(test=True)

#### Run the sample with external H ####

# Add thermostat
system.sys.thermostat.set_langevin(kT=system.kT, gamma=10, seed=system.seed)

# Add magnetic dipole interactions - direct sum, non-preiodic in z
system.sys.periodicity = [True, True, False]
dds = espressomd.magnetostatics.DipolarDirectSumCpu(prefactor=1)
system.sys.magnetostatics.solver = dds
system.sys.integrator.run(0)

# Mark particles to magnetize. Careful to use python lists, and not espressomd particle slices
parts_to_magnetize= []
if any(isinstance(obj, PointDipoleSuperpara) for obj in system.objects):
    pds_to_magnetize = list(system.sys.part.select(type=system.part_types['pds_virt']))
    parts_to_magnetize.append([pds_to_magnetize, config_pds['dipm']])
else:
    parts_to_magnetize = None

# Gradually increasing dipole moments for permanent dipoles
if any(isinstance(obj, PointDipolePermanent) for obj in system.objects):
    system.sys.time_step = 0.001
    dipm_pdp= 0.
    dipm_incr = config_pdp['dipm'] / 10
    system.sys.part.select(type=system.part_types['pdp_real']).dipm = 1E-6 # Cannot start at 0
    system.sys.integrator.run(0, recalc_forces=True)
    for _ in range(10):
        dipm_pdp += dipm_incr
        system.sys.part.select(type=system.part_types['pdp_real']).dipm = dipm_pdp
        for step in range(1):
            system.sys.integrator.run(1, recalc_forces=True)
            for parts, dipm_pds in parts_to_magnetize:
                system.magnetize(part_list=parts, dip_magnitude=dipm_pds, H_ext=np.asarray([0,0,0]))

    assert all(np.abs( np.linalg.norm(system.sys.part.select(type=system.part_types['pdp_real']).dip, axis=-1) - config_pdp['dipm']) < 0.001  )

# STABILIZE MAE WITH MAGNETIC FIELD
system.sys.time = 0.

ext_B_z = sim_params['H']
H_ext = [0,0,ext_B_z]
system.set_H_ext(H=H_ext)

for step in range(2):
    system.sys.integrator.run(1, recalc_forces=True)
    for parts, dipm in parts_to_magnetize:
        system.magnetize(part_list=parts, dip_magnitude=dipm, H_ext=system.get_H_ext())
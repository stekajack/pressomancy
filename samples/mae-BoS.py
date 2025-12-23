import espressomd
import espressomd.magnetostatics
import espressomd.checkpointing
import logging
logging.basicConfig(level=logging.INFO)
from espressomd.io.writer import vtf
                  
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

# sim_inst params
DENS_PART = sim_params['DENS_PART']
MAE_LAYER_HEIGHT_parts = sim_params['HEIGHT'] # in part size units
N_M_FULL_BOX = sim_params['N_FULL_BOX']

SIGMA_PART = 1.
SIZE_PART = SIGMA_PART*pow(2, 1/6)  # in sim_inst units

print(f"SIZE_PART={SIZE_PART}")
R_PART= SIZE_PART/2

BONDS_MAX_LENGHT_A = 5.
BOND_K_A = sim_params['BOND_K']
valid_bond_limits = {'soft': (0.001, 0.01), 'hard': (0.01, 0.1)}
BOND_LIMITS_A = valid_bond_limits.get(BOND_K_A)
if BOND_LIMITS_A is None:
   raise ValueError(f"Invalid bond type: {BOND_K_A}. Valid --K values -> {valid_bond_limits.keys()}")

MAE_LAYER_HEIGHT = MAE_LAYER_HEIGHT_parts * SIZE_PART # in sim_inst units
BOX_SIZE = np.cbrt( N_M_FULL_BOX * 4/3*np.pi / 0.3 ) * R_PART
BOX_Z_MAX = 4 * BOX_SIZE

N_PART = round(DENS_PART * BOX_SIZE**2 * MAE_LAYER_HEIGHT / ( 4/3 * np.pi * R_PART**3))
print(f"N_PART={N_PART}")

assert MAE_LAYER_HEIGHT<=BOX_Z_MAX
assert DENS_PART<=0.3

box_l = [BOX_SIZE, BOX_SIZE, BOX_Z_MAX]
box_E = [BOX_SIZE, BOX_SIZE, MAE_LAYER_HEIGHT]

dens_A = N_PART * 4/3 * np.pi * R_PART**3 / np.prod(box_E)

assert dens_A - DENS_PART < 0.001, f"DENS_{dens_A}"

# INITIALIZE sim_inst
sim_inst = Simulation(box_dim=box_l)
sim_inst.sys.box_l=box_l
sim_inst.seed = sim_params['seed']
sim_inst.set_sys(timestep=0.001)

config_pdp = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)
config_pds = PointDipoleSuperpara.config.specify(dipm=1., espresso_handle=sim_inst.sys)
n_pdp = int(N_PART/2); n_pds = N_PART - n_pdp
associated_objects = [PointDipolePermanent(config=config_pdp) for _ in range(n_pdp)] + [PointDipoleSuperpara(config=config_pds) for _ in range(n_pds)]
assert len(associated_objects) == N_PART
config_E = Elastomer.config.specify(box_E=box_E, n_parts=N_PART, associated_objects=associated_objects, bond_K_lims=BOND_LIMITS_A, size=SIZE_PART, sigma=SIGMA_PART, espresso_handle=sim_inst.sys, seed=sim_inst.seed)
elastomer=[Elastomer(config=config_E) for _ in range(1)]
sim_inst.store_objects(elastomer)
sim_inst.set_objects(elastomer)
elastomer= elastomer[0]

sim_inst.set_steric(key=("pdp_real", "pds_real"))
sim_inst.sys.integrator.run(0)

# must add non_bonded interactions before creating substrate
elastomer.create_substrate(geometry='part')
energy = sim_inst.sys.analysis.energy()
print("total",energy["total"])
print("bonded",energy["bonded"])
print("non_bonded",energy["non_bonded"])

elastomer.mix_elastomer_stuff(test=True)
elastomer.cure_elastomer(test=True)

#### Run the sample with external H ####

# Add thermostat
sim_inst.sys.thermostat.set_langevin(kT=1., gamma=1., seed=sim_inst.seed)

# Add magnetic dipole interactions - direct sum, non-preiodic in z
sim_inst.sys.periodicity = [True, True, False]
dds = espressomd.magnetostatics.DipolarDirectSumCpu(prefactor=1)
sim_inst.sys.magnetostatics.solver = dds
sim_inst.sys.integrator.run(0)

# Mark particles to magnetize. Careful to use python lists, and not espressomd particle slices
pds_to_magnetize = list(sim_inst.sys.part.select(type=sim_inst.part_types['pds_virt']))
parts_to_magnetize = [[pds_to_magnetize, config_pds['dipm']],]

# Gradually increasing dipole moments for permanent dipoles
sim_inst.sys.time_step = 0.001
for _ in range(10):
   sim_inst.sys.integrator.run(1, recalc_forces=True)
   for parts, dipm_pds in parts_to_magnetize:
         sim_inst.magnetize(part_list=parts, dip_magnitude=dipm_pds, H_ext=np.asarray([0,0,1e-6]))


# STABILIZE MAE WITH MAGNETIC FIELD
sim_inst.sys.time = 0.

ext_B_z = sim_params['H']
H_ext = [0,0,ext_B_z]
sim_inst.set_H_ext(H=H_ext)

for step in range(2):
   sim_inst.sys.integrator.run(1, recalc_forces=True)
   for parts, dipm in parts_to_magnetize:
      sim_inst.magnetize(part_list=parts, dip_magnitude=dipm, H_ext=sim_inst.get_H_ext())

#################
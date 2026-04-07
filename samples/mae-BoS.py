import espressomd
import espressomd.version
if espressomd.version.major()==5:
    from espressomd.magnetostatics import DipolarDirectSum
elif espressomd.version.major()==4:
    from espressomd.magnetostatics import DipolarDirectSumCpu
else:
    raise ImportError(f"Unsupported ESPResSo version: {espressomd.version}. Please use version 4 or 5.")
import logging
logging.basicConfig(level=logging.INFO)
                  
espressomd.assert_features(['WCA', 'ROTATION', 'DIPOLES', 'DP3M',
                            'VIRTUAL_SITES', 'VIRTUAL_SITES_RELATIVE',
                            'EXTERNAL_FORCES'])

from pressomancy.simulation import Simulation, Elastomer, PointDipolePermanent, PointDipoleSuperpara

import numpy as np

#################

# Simulation parameters
sim_params = {'DENS_PART': 0.3, 'BOND_K': "hard", 'HEIGHT': 6,
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

dens_A = N_PART * 4/3 * np.pi * R_PART**3 / np.prod([BOX_SIZE, BOX_SIZE, MAE_LAYER_HEIGHT])

assert dens_A - DENS_PART < 0.001, f"DENS_{dens_A}"

# INITIALIZE sim_inst
sim_inst = Simulation(box_dim=box_l)
sim_inst.reinitialize_instance()

sim_inst.sys.box_l=box_l
sim_inst.seed = sim_params['seed']
sim_inst.set_sys(timestep=0.001)

if espressomd.version.major()==5:
    config_pdp = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)
    config_pds = PointDipoleSuperpara.config.specify(dipm=1., Xi_0=0.1, espresso_handle=sim_inst.sys)
    n_pdp = int(N_PART/2); n_pds = N_PART - n_pdp
    associated_objects = [PointDipolePermanent(config=config_pdp) for _ in range(n_pdp)] + [PointDipoleSuperpara(config=config_pds) for _ in range(n_pds)]
elif espressomd.version.major()==4:
    config_pdp = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)
    n_pdp = N_PART
    associated_objects = [PointDipolePermanent(config=config_pdp) for _ in range(n_pdp)]
assert len(associated_objects) == N_PART
config_E = Elastomer.config.specify(layer_height=MAE_LAYER_HEIGHT, n_parts=N_PART, associated_objects=associated_objects, bond_K_lims=BOND_LIMITS_A, size=SIZE_PART, sigma=SIGMA_PART, espresso_handle=sim_inst.sys, seed=sim_inst.seed)
elastomer=[Elastomer(config=config_E) for _ in range(1)]
sim_inst.store_objects(elastomer)
sim_inst.set_objects(elastomer)
elastomer= elastomer[0]

if espressomd.version.major()==5:
    sim_inst.set_steric(key=("pdp_real", "pds_real"), sigma=SIGMA_PART)
if espressomd.version.major()==4:
    sim_inst.set_steric(key=("pdp_real",), sigma=SIGMA_PART)
sim_inst.sys.integrator.run(0)

# must add non_bonded interactions before creating substrate
energy = sim_inst.sys.analysis.energy()
print("total",energy["total"])
print("bonded",energy["bonded"])
print("non_bonded",energy["non_bonded"])

elastomer.mix_elastomer_stuff(test=True)
elastomer.cure_elastomer()

#### Run the sample with external H ####

# Add thermostat
sim_inst.sys.thermostat.set_langevin(kT=1e-3, gamma=10, seed=sim_inst.seed)

# Add magnetic dipole interactions - direct sum, non-preiodic in z
sim_inst.sys.periodicity = [True, True, False]
if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))
sim_inst.sys.integrator.run(0)

# STABILIZE MAE WITH MAGNETIC FIELD
sim_inst.sys.time = 0.

ext_B_z = sim_params['H']
H_ext = [0,0,ext_B_z]
sim_inst.set_H_ext(H=H_ext)

sim_inst.sys.integrator.run(10)

#################
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
                            'EXTERNAL_FORCES', 'MAGNETIZE'])

from pressomancy.simulation import Simulation, PointDipolePermanent, PointDipoleSuperpara

import numpy as np

#################

box_l = [10,10,10]
periodicity = [False, False, False]
H = 1

HM_dipm = 1.

SM_dipm = 1.
SM_Xi_0 = 0.1

# INITIALIZE sim_inst
sim_inst = Simulation(box_dim=box_l)
sim_inst.sys.box_l=box_l
sim_inst.seed = 1
sim_inst.set_sys(timestep=0.1)

pos = np.array([[0.,0.,0.],
                [0.,0.,1.]])
dip = np.array([[0.,0.,1.],
                [0.,0.,1.]])

config_pdp = PointDipolePermanent.config.specify(dipm=HM_dipm, espresso_handle=sim_inst.sys)
config_pds = PointDipoleSuperpara.config.specify(dipm=SM_dipm, Xi_0=SM_Xi_0, mag_func=0, espresso_handle=sim_inst.sys)

# Test one permanent point dipole with external H
pdp_list = [PointDipolePermanent(config=config_pdp) for _ in range(1)]
sim_inst.store_objects(pdp_list)
sim_inst.set_objects(pdp_list) # set with random positions

# put in the right position and dipoles
i = 0
for part in sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"]):
    part.pos = pos[i]
    part.dip = dip[i]
    i+=1

if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))
sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0,0,H])

sim_inst.sys.integrator.run(2)

assert np.array_equal(sim_inst.sys.part.all().pos,pos[:1]), f"{sim_inst.sys.part.all().pos},\n{pos[:1]}"
assert np.array_equal(sim_inst.sys.part.all().dip, dip[:1]), f"{sim_inst.sys.part.all().dip},\n{dip[:1]}"

sim_inst.sys.part.clear()

# Test one superparamagnetic point dipole with external H
pds_list = [PointDipoleSuperpara(config=config_pds) for _ in range(1)]
sim_inst.store_objects(pds_list)
sim_inst.set_objects(pds_list) # set with random positions

# put in the right position with 0 dipole
i = 0
for part in sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]):
    part_real = sim_inst.sys.part.by_id(part.vs_relative[0])
    part_real.pos = pos[i]
    part.pos = pos[i]
    i+=1

if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))
sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0,0,H])

sim_inst.sys.integrator.run(2)

dip_assert = np.array(dip[:1], copy=True)
dip_assert[:,2] = 0.0994051

assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos, pos[:1]), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos},\n{pos[:1]}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos, pos[:1]), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos},\n{pos[:1]}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip, dip[:1]*0.), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip},\n{dip[:1]*0.}"
assert np.allclose(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip, dip_assert), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip},\n{dip_assert}"

sim_inst.sys.part.clear()

# Test two permanent point dipoles aligned in the same direction as the field
pdp_list = [PointDipolePermanent(config=config_pdp) for _ in range(2)]
sim_inst.store_objects(pdp_list)
sim_inst.set_objects(pdp_list) # set with random positions

# put in the right position and dipoles
i = 0
for part in sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"]):
    part.pos = pos[i]
    part.dip = dip[i]
    i+=1

# Fix because point dipoles will atract eachother without volume exclusion
part_list = list(sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"]))
for part in part_list:
    part.fix = [True, True, True]

if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))
sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0,0,H])

sim_inst.sys.integrator.run(2)

assert np.array_equal(sim_inst.sys.part.all().pos, pos), f"{sim_inst.sys.part.all().pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.all().dip, dip), f"{sim_inst.sys.part.all().dip},\n{dip}"

#test dipolar field
dip_fld = np.array([[0,0,2],
                    [0,0,2]])
assert np.array_equal(sim_inst.sys.part.all().dip_fld, dip_fld), f"{sim_inst.sys.part.all().dip_fld},\n{dip_fld}"

sim_inst.sys.part.clear()

# Test two superparamagnetic point dipoles aligned in the same direction as the field (Langevin magnetization)
pds_list = [PointDipoleSuperpara(config=config_pds) for _ in range(2)]
sim_inst.store_objects(pds_list)
sim_inst.set_objects(pds_list) # set with random positions

# put in the right position with 0 dipole
i = 0
for part in sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]):
    part_real = sim_inst.sys.part.by_id(part.vs_relative[0])
    part_real.pos = pos[i]
    part.pos = pos[i]
    i+=1

# Fix because point dipoles will atract eachother wihout volume exclusion
part_list = list(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]))
for part in part_list:
    part.fix = [True, True, True]

if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))
sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0,0,H])

sim_inst.sys.integrator.run(10)


dip_assert = np.array(dip, copy=True)
dip_assert[:,2] = 0.12356435

#NOTE FOR TEST DEBUGGING. If the problem is the dipole momento modulus, try to increase the number of iterations by 1, until it works (probably just one more will do the trick). If the values are very close, something inside espresso might have changed this value.

assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip, dip*0.), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip},\n{dip*0.}"
assert np.allclose(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip, dip_assert, atol=0.00005, rtol=0), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip},\n{dip_assert}"

poss_for_next_test = sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos
dips_for_next_test = sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip

sim_inst.sys.part.clear()

# Test two superparamagnetic point dipoles aligned in the same direction as the field (Langevin magnetization) with higher susceptibylity - same as pressomancy magnetize
config_pds = PointDipoleSuperpara.config.specify(dipm=SM_dipm, Xi_0=1./3., mag_func=0, espresso_handle=sim_inst.sys)
pds_list = [PointDipoleSuperpara(config=config_pds) for _ in range(2)]
sim_inst.store_objects(pds_list)
sim_inst.set_objects(pds_list) # set with random positions

# put in the right position with 0 dipole
i = 0
for part in sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]):
    part_real = sim_inst.sys.part.by_id(part.vs_relative[0])
    part_real.pos = pos[i]
    part.pos = pos[i]
    i+=1

# Fix because point dipoles will atract eachother wihout volume exclusion
part_list = list(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]))
for part in part_list:
    part.fix = [True, True, True]

if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))
sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0,0,H])

sim_inst.sys.integrator.run(10)


dip_assert = np.array(dip, copy=True)
dip_assert[:,2] = 0.55634

#NOTE FOR TEST DEBUGGING. If the problem is the dipole momento modulus, try to increase the number of iterations by 1, until it works (probably just one more will do the trick). If the values are very close, something inside espresso might have changed this value.

assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip, dip*0.), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip},\n{dip*0.}"
assert np.allclose(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip, dip_assert, atol=0.00005, rtol=0), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip},\n{dip_assert}"

poss_for_next_test = sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos
dips_for_next_test = sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip

sim_inst.sys.part.clear()

# Test two superparamagnetic point dipoles aligned in the same direction as the field (Langevin magnetization) - for pressomancy magnetize
pds_list = [PointDipoleSuperpara(config=config_pds) for _ in range(2)]
sim_inst.store_objects(pds_list)
sim_inst.set_objects(pds_list) # set with random positions

# put in the right position with 0 dipole
i = 0
for part in sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]):
    part_real = sim_inst.sys.part.by_id(part.vs_relative[0])
    part_real.pos = pos[i]
    part.pos = pos[i]
    i+=1

# Fix because point dipoles will atract eachother wihout volume exclusion
part_list = list(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]))
for part in part_list:
    part.fix = [True, True, True]

if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))
sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0,0,H])

parts_to_magnetize = list(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]))
for part in parts_to_magnetize:
    part.is_magnetizable = False # remove espresso magnetization to compare to pressomancy magnetization

H_ext = sim_inst.get_H_ext()
assert np.array_equal(H_ext, [0,0,H])
sim_inst.magnetize(parts_to_magnetize, SM_dipm, H_ext)
for step in range(10):
    sim_inst.sys.integrator.run(1)
    sim_inst.magnetize(parts_to_magnetize, SM_dipm, H_ext)

dip_assert = np.array(dip, copy=True)
dip_assert[:,2] = 0.55634

#NOTE FOR TEST DEBUGGING. If the problem is the dipole momento modulus, try to increase the number of iterations by 1, until it works (probably just one more will do the trick). If the values are very close, something inside espresso might have changed this value.

assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip, dip*0.), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip},\n{dip*0.}"
assert np.allclose(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip, dip_assert, atol=0.00005, rtol=0), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip},\n{dip_assert}"

assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos, poss_for_next_test), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos},\n{poss_for_next_test}"
assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip, dips_for_next_test), f"{sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip},\n{dips_for_next_test}"

sim_inst.sys.part.clear()
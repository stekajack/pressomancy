import espressomd
import espressomd.version
import numpy as np

from pressomancy.helper_functions import api_agnostic_feature_check
from pressomancy.simulation import Simulation, PointDipolePermanent, PointDipoleSuperpara

if espressomd.version.major() == 5:
    from espressomd.magnetostatics import DipolarDirectSum
elif espressomd.version.major() == 4:
    from espressomd.magnetostatics import DipolarDirectSumCpu
else:
    raise RuntimeError(f"Unsupported ESPResSo version {espressomd.version}")

HAS_SUPERPARA_FEATURES = all(
    api_agnostic_feature_check(feature)
    for feature in PointDipoleSuperpara.required_features
)

box_l = [10, 10, 10]
H = 1

HM_dipm = 1.
SM_dipm = 1.
SM_Xi_0 = 0.1

sim_inst = Simulation(box_dim=box_l)
sim_inst.sys.box_l = box_l
sim_inst.seed = 1
sim_inst.set_sys(timestep=0.1)

pos = np.array([[0., 0., 0.],
                [0., 0., 1.]])
dip = np.array([[0., 0., 1.],
                [0., 0., 1.]])

config_pdp = PointDipolePermanent.config.specify(dipm=HM_dipm, espresso_handle=sim_inst.sys)

# Test two permanent point dipoles aligned in the same direction as the field.
pdp_list = [PointDipolePermanent(config=config_pdp) for _ in range(2)]
sim_inst.store_objects(pdp_list)
sim_inst.set_objects(pdp_list)

for i, part in enumerate(sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"])):
    part.pos = pos[i]
    part.dip = dip[i]
    part.fix = [True, True, True]

sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0, 0, H])
sim_inst.sys.integrator.run(2)

assert np.array_equal(sim_inst.sys.part.all().pos, pos), f"{sim_inst.sys.part.all().pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.all().dip, dip), f"{sim_inst.sys.part.all().dip},\n{dip}"

sim_inst.sys.part.clear()

if HAS_SUPERPARA_FEATURES:
    config_pds = PointDipoleSuperpara.config.specify(
        dipm=SM_dipm,
        Xi_0=SM_Xi_0,
        mag_func=0,
        espresso_handle=sim_inst.sys,
    )

    # Test two superparamagnetic point dipoles aligned in the same direction as the field.
    pds_list = [PointDipoleSuperpara(config=config_pds) for _ in range(2)]
    sim_inst.store_objects(pds_list)
    sim_inst.set_objects(pds_list)

    for i, part in enumerate(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"])):
        part_real = sim_inst.sys.part.by_id(part.vs_relative[0])
        part_real.pos = pos[i]
        part.pos = pos[i]
        part_real.fix = [True, True, True]

    if espressomd.version.major() == 5:
        sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
    else:
        sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))

    sim_inst.sys.integrator.run(0)
    sim_inst.set_H_ext(H=[0, 0, H])
    sim_inst.sys.integrator.run(10)

    dip_assert = np.array(dip, copy=True)
    dip_assert[:, 2] = 0.12356435

    assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types['pds_real']).pos},\n{pos}"
    assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types['pds_virt']).pos},\n{pos}"
    assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]).dip, dip * 0.), f"{sim_inst.sys.part.select(type=sim_inst.part_types['pds_real']).dip},\n{dip * 0.}"
    assert np.allclose(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"]).dip, dip_assert, atol=0.00005, rtol=0), f"{sim_inst.sys.part.select(type=sim_inst.part_types['pds_virt']).dip},\n{dip_assert}"

    poss_for_next_test = sim_inst.sys.part.select(type=sim_inst.part_types['pds_virt']).pos
    dips_for_next_test = sim_inst.sys.part.select(type=sim_inst.part_types['pds_virt']).dip

    sim_inst.sys.part.clear()

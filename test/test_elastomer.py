from pressomancy.simulation import Elastomer, PointDipolePermanent
from create_system import sim_inst, BaseTestCase
import numpy as np
from collections import defaultdict

import espressomd
assert espressomd.version.major() in (4,5)

class ElastomerTest(BaseTestCase):
    box_E = [3,3,9]
    part_size=1.
    conf_magn=PointDipolePermanent.config.specify(
            dipm=1.,espresso_handle=sim_inst.sys)

    def setUp(self) -> None:
        self.mag_part = [PointDipolePermanent(config=self.conf_magn) for _ in range(10)]
        self.instance_cust=Elastomer(config=Elastomer.config.specify(
            box_E=self.box_E, n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed,associated_objects=self.mag_part)
            )

    def tearDown(self) -> None:
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_cure_elastomer(self):
        sim_inst.store_objects(self.mag_part)
        sim_inst.store_objects([self.instance_cust])
        sim_inst.set_objects([self.instance_cust])
        self.instance_cust.cure_elastomer()

        n_bond_dict = defaultdict(list)
        for part in sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"]):
            for bond in part.bonds:
                n_bond_dict[part.id].append(bond[1])
                n_bond_dict[bond[1]].append(part.id)

        n_bonds_per_part_list = [len(bonds) for bonds in n_bond_dict.values()]
        assert min(n_bonds_per_part_list) > 0, f"{n_bonds_per_part_list}"
        dist_bonds_per_part_list = [np.linalg.norm(part.pos - sim_inst.sys.part.by_id(id).pos) for part in sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"]) for bond, id in part.bonds]
        assert max(dist_bonds_per_part_list) <= 5.001, f"{max(dist_bonds_per_part_list)}"

    def test_substrate(self):
        sim_inst.store_objects(self.mag_part)
        sim_inst.store_objects([self.instance_cust])
        sim_inst.set_objects([self.instance_cust])

        substrate_pos = np.asarray([[ 0.5,0.5, 0.5],
                                    [ 1.5,0.5, 0.5],
                                    [ 2.5,0.5, 0.5],
                                    [ 0.5,1.5, 0.5],
                                    [ 1.5,1.5, 0.5],
                                    [ 2.5,1.5, 0.5],
                                    [ 0.5,2.5, 0.5],
                                    [ 1.5,2.5, 0.5],
                                    [ 2.5,2.5, 0.5]])

        assert set(map(tuple, np.asarray([p.pos for p in self.instance_cust.substrate]))) == set(map(tuple, substrate_pos)), f"{[p.pos for p in self.instance_cust.substrate]}"

        self.instance_cust.remove_substrate()

        assert len(sim_inst.sys.part.select(type=sim_inst.part_types["substrate"])) == 0


    def test_mix_elastomer_stuff_restores_langevin_thermostat(self):
        sim_inst.store_objects(self.mag_part)
        sim_inst.store_objects([self.instance_cust])
        sim_inst.set_objects([self.instance_cust])

        sim_inst.sys.thermostat.set_langevin(kT=0.7, gamma=3.5, seed=41)
        self.instance_cust.mix_elastomer_stuff(n_iter=0)

        self.assertFalse(sim_inst.sys.thermostat.call_method("is_off"))
        self.assertTrue(sim_inst.sys.thermostat.langevin.is_active)
        self.assertAlmostEqual(sim_inst.sys.thermostat.kT, 0.7)
        np.testing.assert_allclose(np.copy(sim_inst.sys.thermostat.langevin.gamma), 3.5)
        self.assertEqual(sim_inst.sys.thermostat.langevin.seed, 41)

    def test_snapshot_restore_brownian_thermostat(self):
        sim_inst.sys.thermostat.set_brownian(kT=0.9, gamma=2.5, seed=43)

        snapshot = self.instance_cust._snapshot_thermostat_state()
        sim_inst.sys.thermostat.turn_off()
        self.instance_cust._restore_thermostat_state(snapshot)

        self.assertFalse(sim_inst.sys.thermostat.call_method("is_off"))
        self.assertTrue(sim_inst.sys.thermostat.brownian.is_active)
        self.assertAlmostEqual(sim_inst.sys.thermostat.kT, 0.9)
        np.testing.assert_allclose(np.copy(sim_inst.sys.thermostat.brownian.gamma), 2.5)
        self.assertEqual(sim_inst.sys.thermostat.brownian.seed, 43)

    def test_wall_substrate(self):
        instance_def=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
                n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed))
        sim_inst.store_objects([instance_def])
        sim_inst.set_objects([instance_def])

        instance_def.remove_substrate()
        instance_def.create_substrate(geometry="wall")
        instance_def.cure_elastomer()

        n_bond_dict = defaultdict(list)
        for part in sim_inst.sys.part.select(type=sim_inst.part_types["real"]):
            for bond in part.bonds:
                n_bond_dict[part.id].append(bond[1])
                n_bond_dict[bond[1]].append(part.id)

        n_bonds_per_part_list = [len(bonds) for bonds in n_bond_dict.values()]
        assert min(n_bonds_per_part_list) > 0, f"{n_bonds_per_part_list}"

    def test_bond_to_neighbors(self):
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=4, size=self.part_size,
            espresso_handle=sim_inst.sys,seed=sim_inst.seed))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        positions = np.asarray([[0.90, 1.00, 1.6],
                                [1.10, 1.00, 1.6],
                                [1.00, 0.90, 1.6],
                                [1.00, 1.10, 1.6]])
        for part, pos in zip(instance.type_part_dict["real"], positions):
            part.pos = pos

        instance.bond_to_neighbors(parts=sim_inst.sys.part.select(type=sim_inst.part_types["real"]), n_nghb=3, bond_k=0.05, r_catch=0.5)

        n_bond_dict = defaultdict(list)
        for part in sim_inst.sys.part.select(type=sim_inst.part_types["real"]):
            for bond in part.bonds:
                n_bond_dict[part.id].append(bond[1])
                n_bond_dict[bond[1]].append(part.id)

        n_bonds_per_part_list = [len(bonds) for bonds in n_bond_dict.values()]
        assert min(n_bonds_per_part_list) >= 3, f"{n_bonds_per_part_list}"

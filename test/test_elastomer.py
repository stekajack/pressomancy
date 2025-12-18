from pressomancy.simulation import Elastomer, PointDipoleSuperpara
from create_system import sim_inst, BaseTestCase
import numpy as np

class ElastomerTest(BaseTestCase):
    box_E = [3,3,9]

    part_size=1.

    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_set_object_generic(self):
        instance=Elastomer(config=Elastomer.config.specify(
            n_parts=10, size=self.part_size, espresso_handle=sim_inst.sys, seed=sim_inst.seed))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

    def test_set_object(self):
        mag_part = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=1.,
            espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed,
            associated_objects=mag_part))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

    def test_mix_elastomer(self):
        mag_part = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=1.,
            espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed,
            associated_objects=mag_part))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        sim_inst.set_steric(tuple((type_name for type_name in instance.part_types.keys())))

        instance.mix_elastomer_stuff(test=True)

    def test_cure_elastomer(self):
        from collections import defaultdict
        mag_part = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=1.,
            espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed,
            associated_objects=mag_part))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        instance.cure_elastomer(test=True)

        n_bond_dict = defaultdict(list)
        for part in sim_inst.sys.part.select(type=62):
            for bond in part.bonds:
                n_bond_dict[part.id].append(bond[1])
                n_bond_dict[bond[1]].append(part.id)

        n_bonds_per_part_list = [len(bonds) for bonds in n_bond_dict.values()]
        assert min(n_bonds_per_part_list) > 0, f"{n_bonds_per_part_list}"
        dist_bonds_per_part_list = [np.linalg.norm(part.pos - sim_inst.sys.part.by_id(id).pos) for part in sim_inst.sys.part.select(type=62) for bond, id in part.bonds]
        assert max(dist_bonds_per_part_list) <= 5.

    def test_cure_bad_elastomer(self):
        from collections import defaultdict
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,
            bond_cutoff=0.9,
            espresso_handle=sim_inst.sys,seed=sim_inst.seed))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        instance.cure_elastomer(test=True, test_bad=True)

        n_bond_dict = defaultdict(list)
        for part in sim_inst.sys.part.select(type=sim_inst.part_types["real"]):
            for bond in part.bonds:
                n_bond_dict[part.id].append(bond[1])
                n_bond_dict[bond[1]].append(part.id)

        n_bonds_per_part_list = [len(bonds) for bonds in n_bond_dict.values()]
        assert min(n_bonds_per_part_list) >= 3, f"{n_bonds_per_part_list}"

    def test_substrate(self):
        mag_part = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=1.,
            espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed,
            associated_objects=mag_part))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        instance.create_substrate()

        substrate_pos = np.asarray([[ 0.5,0.5,-0.5],
                                    [ 1.5,0.5,-0.5],
                                    [ 2.5,0.5,-0.5],
                                    [ 0.5,1.5,-0.5],
                                    [ 1.5,1.5,-0.5],
                                    [ 2.5,1.5,-0.5],
                                    [ 0.5,2.5,-0.5],
                                    [ 1.5,2.5,-0.5],
                                    [ 2.5,2.5,-0.5]])

        assert set(map(tuple, np.asarray([p.pos for p in instance.substrate]))) == set(map(tuple, substrate_pos))
    
    def test_remove_substrate(self):
        mag_part = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=1.,
            espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed,
            associated_objects=mag_part))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        instance.create_substrate()

        instance.remove_substrate()

        assert len(sim_inst.sys.part.select(type=sim_inst.part_types["substrate"])) == 0

    def test_elastomer_with_substrate(self):
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        instance.create_substrate()

        instance.mix_elastomer_stuff(test=True)

        instance.cure_elastomer(test=True)

        assert (np.asarray(sim_inst.sys.part.select(type=sim_inst.part_types["real"]).pos)[:,2] >= 0.5).all()

    def test_wall_substrate(self):
        mag_part = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=1.,
            espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        instance=Elastomer(config=Elastomer.config.specify(box_E=self.box_E,
            n_parts=10, size=self.part_size,espresso_handle=sim_inst.sys,seed=sim_inst.seed,
            associated_objects=mag_part))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        instance.create_substrate(geometry="wall")

        instance.remove_substrate(geometry="wall")
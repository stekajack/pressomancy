from pressomancy.simulation import partition_cuboid_volume, get_neighbours, EspressoPart
from test.create_system import sim_inst , BaseTestCase
import numpy as np

class RandomBondsTest(BaseTestCase):
    num_part=5

    sph_diam=1
    sph_rad=0.5*sph_diam

    r_catch=sph_diam

    def setUp(self):
        sim_inst.box_l=np.array([2.5,2.5,2.5])

    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_general(self):
        pos, _,_ = partition_cuboid_volume(sim_inst.box_l,self.num_part,self.sph_diam, flag='norand')
        instance=[EspressoPart(config=EspressoPart.config.specify(espresso_handle=sim_inst.sys)) for x in range(len(pos))]
        sim_inst.store_objects(instance)
        sim_inst.place_objects(instance, pos)

        n_bonds, n_bonds_dict = sim_inst.random_harmonic_bonds(self.r_catch)

        assert n_bonds <= (self.num_part * (self.num_part-1))
        for bonds in sim_inst.sys.part.all().bonds:
            assert all([bond[0].r_0 <= self.r_catch for bond in bonds])
            assert all([(bond[0].k <= 0.01 and bond[0].k >= 0.001) for bond in bonds])

        neighbours_dict = get_neighbours(pos,sim_inst.box_l[0],cuttoff=self.r_catch)

        neighbours_count = {}
        for particle in sim_inst.sys.part.all():
            neighbours_count[particle.id]=0
            for id2 in neighbours_dict[particle.id]:
                tmp_1=sum(1 for _, id_ in sim_inst.sys.part.by_id(id2).bonds if id_==particle.id)
                if tmp_1 == 1:
                    neighbours_count[particle.id] += 1
                
                tmp_2 = sum(1 for _, id_ in particle.bonds if id_==id2)
                if tmp_2 == 1:
                    neighbours_count[particle.id] += 1
                
                tmp_3 = tmp_1 + tmp_2
                if tmp_3 == 0:
                    raise ValueError(f"{id2} not bounded to {particle.id}")
                elif tmp_3 > 1:
                    raise ValueError(f"{id2} and {particle.id} are bounded more than once.")


    # def test_max_bonds(self):
    #     pos, _,_ = partition_cuboid_volume(self.box_len,self.num_vol_side,self.sph_diam, flag='norand')
    #     n_bonds, n_bonds_dict = sim_inst.random_harmonic_bonds(self.r_catch, max_bonds=1)

    #     assert n_bonds <= self.num_part
    #     for particle in sim_inst.sys.part.all():
    #         assert len(particle.bonds) <= 1

    # def test_object_types(self):
    #     pos, _,_ = partition_cuboid_volume(self.box_len,self.num_vol_side,self.sph_diam, flag='norand')
    #     n_bonds, n_bonds_dict = sim_inst.random_harmonic_bonds(self.r_catch, object_types=None)

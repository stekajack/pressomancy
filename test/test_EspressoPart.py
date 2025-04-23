from pressomancy.object_classes import EspressoPart
from pressomancy.helper_functions import generate_positions
from test.create_system import sim_inst , BaseTestCase
import espressomd
import numpy as np

class EspressoPartTest(BaseTestCase):

    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_set_object_generic(self):
        instance=EspressoPart(config=EspressoPart.config.specify(espresso_handle=sim_inst.sys))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])
    
    def test_set_object_dip(self):
        instance=EspressoPart(config=EspressoPart.config.specify(dipm=3., rotation=[True,True,True],
                espresso_handle=sim_inst.sys))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        for particle in sim_inst.sys.part.all():
            assert np.isclose(np.linalg.norm(particle.dip), 3.)

    def test_set_object_director(self):
        instance=EspressoPart(config=EspressoPart.config.specify(rotation=[True,True,True],
                espresso_handle=sim_inst.sys))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        for particle in sim_inst.sys.part.all():
            assert np.isclose(np.linalg.norm(particle.director), 1.)

    def test_place_object_generic(self):
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)
        instance=[EspressoPart(config=EspressoPart.config.specify(espresso_handle=sim_inst.sys)) for x in range(4)]
        sim_inst.store_objects(instance)
        pos = generate_positions(4, 2.5, 1)
        sim_inst.place_objects(instance, pos)

        particle_pos_list = sim_inst.sys.part.all().pos
        for i in range(len(particle_pos_list)):
            for j in range(i+1,len(particle_pos_list)):
                self.assertGreaterEqual(abs(np.linalg.norm(particle_pos_list[i] - particle_pos_list[j])), 1)

            np.testing.assert_allclose(pos[i], particle_pos_list[i])

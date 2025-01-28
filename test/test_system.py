import pressomancy.object_classes
import inspect
import logging
from create_system import sim_inst, BaseTestCase
from pressomancy.helper_functions import MissingFeature
import espressomd

class SimulationTest(BaseTestCase):
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam
    
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()

    def test_store_set_del_objects(self):
        classes = [member for name, member in inspect.getmembers(pressomancy.object_classes, inspect.isclass)]
        for cls in classes:
            try:
                instance=[cls(config=cls.config.specify(espresso_handle=sim_inst.sys)),]
                sim_inst.store_objects(instance)
                sim_inst.set_objects(instance)
                instance[0].delete_owned_parts()
            except MissingFeature:
                logging.warning(f"Skipping {cls.__name__} because it requires a feature that is not available.")
                continue
            self.assertEqual(len(sim_inst.sys.part), 0)
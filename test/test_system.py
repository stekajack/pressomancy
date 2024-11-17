import numpy as np
import unittest
import pressomancy.object_classes
import inspect
from create_system import sim_inst


class SimulationTest(unittest.TestCase):
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam
    
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()


    def create_instance(self, cls):
        """Helper to create an instance of a given class with default arguments."""
        try:
            return [cls(espresso_handle=sim_inst.sys, sigma=1.)]
        except TypeError as e:
            print(f"Skipped test for {cls.__name__}. Instantiation raised {e}")
            return None
    
    def test_store_objects(self):
        classes = [member for name, member in inspect.getmembers(pressomancy.object_classes, inspect.isclass)]
        for cls in classes:
            instance=self.create_instance(cls)
            if instance:
                sim_inst.store_objects(instance)

    def test_set_objects(self):
        classes = [member for name, member in inspect.getmembers(pressomancy.object_classes, inspect.isclass)]
        for cls in classes:
            instance=self.create_instance(cls)
            if instance:
                sim_inst.store_objects(instance)
                sim_inst.set_objects(instance)
    
    def test_store_quadriplex(self):
    
        quartets = [pressomancy.object_classes.Quartet(sigma=1., n_parts=25, espresso_handle=sim_inst.sys) for x in range(3)]
        instance=[pressomancy.object_classes.Quadriplex(sigma=1.,espresso_handle=sim_inst.sys,quartet_grp=quartets),]
        sim_inst.store_objects(instance)
    
    def test_set_quadriplex(self):
    
        quartets = [pressomancy.object_classes.Quartet(sigma=1., n_parts=25, espresso_handle=sim_inst.sys) for x in range(3)]
        sim_inst.store_objects(quartets)
        instance=[pressomancy.object_classes.Quadriplex(sigma=1.,espresso_handle=sim_inst.sys,quartet_grp=quartets),]
        sim_inst.store_objects(instance)
        sim_inst.set_objects(instance)
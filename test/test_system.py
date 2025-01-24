import pressomancy.object_classes
import inspect
import logging
from create_system import sim_inst, BaseTestCase
import espressomd

class SimulationTest(BaseTestCase):
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
            logging.warning(f"Skipped test for {cls.__name__}. Instantiation raised {e}")
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
        bond_hndl=espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=1.5*2)
        sim_inst.sys.bonded_inter.add(bond_hndl)
        instance=[pressomancy.object_classes.Quadriplex(sigma=1.,espresso_handle=sim_inst.sys,quartet_grp=quartets,bonding_mode='ftf',bond_handle=bond_hndl),]
        sim_inst.store_objects(instance)
    
    def test_set_quadriplex(self):
    
        quartets = [pressomancy.object_classes.Quartet(sigma=1., n_parts=25, espresso_handle=sim_inst.sys) for x in range(3)]
        sim_inst.store_objects(quartets)
        bond_hndl=espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=1.5*2)
        sim_inst.sys.bonded_inter.add(bond_hndl)
        instance=[pressomancy.object_classes.Quadriplex(sigma=1.,espresso_handle=sim_inst.sys,quartet_grp=quartets,bonding_mode='ftf',bond_handle=bond_hndl),]
        sim_inst.store_objects(instance)
        sim_inst.set_objects(instance)

    def test_del_own_part(self):
        quartets = [pressomancy.object_classes.Quartet(sigma=1., n_parts=25, espresso_handle=sim_inst.sys) for x in range(3)]
        sim_inst.store_objects(quartets)
        sim_inst.set_objects(quartets)
        for obj in quartets:
            obj.delete_owned_parts()
        self.assertEqual(len(sim_inst.sys.part), 0)
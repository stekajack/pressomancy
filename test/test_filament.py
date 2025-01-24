import pressomancy.object_classes
from create_system import sim_inst , BaseTestCase
import espressomd

class FilamentTest(BaseTestCase):
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam

    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_set_object_generic(self):
        instance=pressomancy.object_classes.Filament(sigma=1, n_parts=10,espresso_handle=sim_inst.sys)
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])
    
    def test_set_object(self):
        quartets = [pressomancy.object_classes.Quartet(sigma=1., n_parts=25, espresso_handle=sim_inst.sys) for x in range(10)]
        sim_inst.store_objects(quartets)
        instance=pressomancy.object_classes.Filament(sigma=1, n_parts=10,espresso_handle=sim_inst.sys,associated_objects=quartets)
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])
    
    def test_bond_center_to_center(self):
        instance=pressomancy.object_classes.Filament(sigma=1, n_parts=10,espresso_handle=sim_inst.sys)
        bond_hndl=espressomd.interactions.FeneBond(k=10, d_r_max=3, r_0=0)
        instance.bond_center_to_center(bond_handle=bond_hndl,type_key='real')
        quartets = [pressomancy.object_classes.Quartet(sigma=1., n_parts=25, espresso_handle=sim_inst.sys) for x in range(10)]
        instance=pressomancy.object_classes.Filament(sigma=1, n_parts=10,espresso_handle=sim_inst.sys,associated_objects=quartets)
        instance.bond_center_to_center(bond_handle=bond_hndl,type_key='real')

    # def test_add_patches_triples(self):
    #     self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
    #     self.instance.add_patches_triples()
    #     self.instance.delete_object()

    
    # def test_delete(self):
    #     self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
    #     self.instance.delete_object()
    #     self.assertEqual(len(sim_inst.sys.part),0)

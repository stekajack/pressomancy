from pressomancy.object_classes import Filament, Quartet
from create_system import sim_inst, BaseTestCase
import espressomd

class FilamentTest(BaseTestCase):
    num_vol_all=14
    num_vol_side=5
    size_filament=11.3

    sph_diam=1
    sph_rad=0.5*sph_diam

    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_set_object_generic(self):
        instance=Filament(config=Filament.config.specify(
            n_parts=10, size=self.size_filament,espresso_handle=sim_inst.sys))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])
    
    def test_set_object(self):
        quartets = [Quartet(config=Quartet.config.specify(
            espresso_handle=sim_inst.sys)) for x in range(10)]
        sim_inst.store_objects(quartets)
        instance=Filament(config=Filament.config.specify(
            n_parts=10, size=self.size_filament,espresso_handle=sim_inst.sys,associated_objects=quartets))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])
    
    def test_bond_center_to_center(self):
        instance=Filament(config=Filament.config.specify(
            n_parts=10, size=self.size_filament,espresso_handle=sim_inst.sys))
        instance.bond_center_to_center(type_name='real')
        quartets = [Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys)) for x in range(10)]
        instance=Filament(config=Filament.config.specify(
            n_parts=10,size=self.size_filament,espresso_handle=sim_inst.sys,associated_objects=quartets))
        instance.bond_center_to_center(type_name='real')

    def test_bending_potential(self):
        
        instance=Filament(config=Filament.config.specify(
            n_parts=10, size=self.size_filament, espresso_handle=sim_inst.sys)) 
        angle_harmonic=espressomd.interactions.AngleHarmonic(bend=1., phi0=3.)
        instance.add_bending_potential(type_name='real',bond_handle=angle_harmonic)
        quartets = [Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys)) for x in range(10)]
        instance=Filament(config=Filament.config.specify(
            n_parts=10, size=self.size_filament, espresso_handle=sim_inst.sys,associated_objects=quartets))
        instance.add_bending_potential(type_name='real',bond_handle=angle_harmonic)

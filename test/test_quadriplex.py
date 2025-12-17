import numpy as np
import pressomancy.object_classes
import espressomd
from create_system import sim_inst, BaseTestCase
from pressomancy.object_classes.quadriplex_class import *



class QuadriplexTest(BaseTestCase):
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam

    def setUp(self) -> None:
        self.instance=Quadriplex(config=Quadriplex.config.specify(espresso_handle=sim_inst.sys))
        sim_inst.store_objects([self.instance,])
        
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_set_object(self):
        self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        quartets=self.instance.params['associated_objects']
        flattened_parts=[]
        for quartet in quartets:
            parts,_=quartet.get_owned_part()
            flattened_parts.extend(parts)
        check_num=quartets[0].config['n_parts']-1
        for p in flattened_parts:
            self.assertEqual(len(p.exclusions), check_num)
        
    def test_add_patches_triples(self):
        self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance.add_patches_triples()

    # def test_bond_quartets_center_to_center(self):
    
    # def test_bond_quartets_corner_to_corner(self):
    
    def test_add_bending_potential(self):
        self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))

        int_nhdl=espressomd.interactions.AngleHarmonic(bend=1, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(int_nhdl)
        self.instance.add_bending_potential(bending_potential_handle=int_nhdl)
        instance=Quadriplex(config=Quadriplex.config.specify(espresso_handle=sim_inst.sys, bonding_mode='ctc'))
        sim_inst.store_objects([instance,])
        instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        instance.add_bending_potential(bending_potential_handle=int_nhdl)
    
    def test_set_object_broken(self):
        quartets = [Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys,type='broken')) for x in range(3)]
        sim_inst.store_objects(quartets)
        instance=Quadriplex(Quadriplex.config.specify(espresso_handle=sim_inst.sys, associated_objects=quartets, bonding_mode='ctc'))
        sim_inst.store_objects([instance,])
        instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))

    # def test_mark_covalent_bonds(self):


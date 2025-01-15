import numpy as np
import pressomancy.object_classes
import espressomd
from create_system import sim_inst , BaseTestCase



class QuadriplexTest(BaseTestCase):
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam

    def setUp(self) -> None:
        self.quartets = [pressomancy.object_classes.Quartet(sigma=1., n_parts=25, espresso_handle=sim_inst.sys) for x in range(3)]
        sim_inst.store_objects(self.quartets)
        self.instance=pressomancy.object_classes.Quadriplex(sigma=1.,espresso_handle=sim_inst.sys,quartet_grp=self.quartets)
        sim_inst.store_objects([self.instance,])
        
    def tearDown(self) -> None:
        self.quartets=None
        self.instance=None
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_set_object(self):
        self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        
    def test_add_patches_triples(self):
        self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance.add_patches_triples()

    # def test_bond_quartets_center_to_center(self):
    
    # def test_bond_quartets_corner_to_corner(self):
    
    def test_add_bending_potential(self):
        self.instance.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))

        self.instance.__class__.bonding_mode='ctc'
        int_nhdl=espressomd.interactions.AngleHarmonic(bend=1, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(int_nhdl)
        self.instance.__class__.bending_handle=int_nhdl
        self.instance.add_bending_potential()
        self.instance.__class__.bonding_mode='ftf'
        self.instance.add_bending_potential()


    # def test_mark_covalent_bonds(self):


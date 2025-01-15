from pressomancy.simulation import partition_cubic_volume, get_neighbours, get_neighbours_cross_lattice
from create_system import BaseTestCase

class PartitioningTest(BaseTestCase):
    box_len=2.5
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam

    def test_get_neighbours(self):
        control= {0: [2,], 1: [2,],2: [0, 1, 3, 4, ], 3: [2, ], 4: [2,]}
        sphere_centers_short, _,_=partition_cubic_volume(self.box_len,self.num_vol_side,self.sph_diam, flag='norand')
        neigh=get_neighbours(sphere_centers_short,self.box_len,cuttoff=self.sph_diam)
        self.assertEqual(neigh,control,'the get_neighbour method failed to reproduce correct neighbour pairs for a single face of an fcc lattice')

    def test_get_neighbours_cross_lattice(self):
        
        control={0: [0, 2, 5, 6], 1: [1, 2, 5, 7], 2: [0, 1, 2, 3, 4, 5, 6, 7, 8], 3: [2, 3, 6, 8], 4: [2, 4, 7, 8]}
        sphere_centers_long, _,_=partition_cubic_volume(self.box_len,self.num_vol_all,self.sph_diam,flag='norand')

        sphere_centers_short, _,_=partition_cubic_volume(self.box_len,self.num_vol_side,self.sph_diam,flag='norand')

        neigh=get_neighbours_cross_lattice(sphere_centers_short,sphere_centers_long,self.box_len,cuttoff=self.sph_diam)
        self.assertEqual(neigh,control,'the get_neighbour method failed to reproduce correct neighbour pairs for a single face of an fcc lattice')  
        
# control={0: [0, 1, 2, 3, 5, 6, 9],
        #      1: [ 0,  1,  2,  4,  5,  7, 10],
        #      2: [ 0,  1,  2,  3,  4,  5,  6,  7,  8, 11],
        #      3: [ 0,  2,  3,  4,  6,  8, 12],
        #      4:[ 1,  2,  3,  4,  7,  8, 13]}
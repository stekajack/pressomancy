import numpy as np

from pressomancy.simulation import partition_cubic_volume, get_neighbours, get_neighbours_cross_lattice
from create_system import BaseTestCase

class PartitioningTest(BaseTestCase):
    box_len=np.array([2.5, 2.5, 2.5])
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam
    rect_box = np.array([10.0, 20.0, 30.0])

    def test_get_neighbours(self):
        control= {0: [2,], 1: [2,],2: [0, 1, 3, 4,], 3: [2, ], 4: [2,]}
        sphere_centers_short, _,_=partition_cubic_volume(np.ones(3) * self.box_len,self.num_vol_side,self.sph_diam, flag='norand')
        neigh=get_neighbours(sphere_centers_short,np.ones(3) * self.box_len,cuttoff=self.sph_diam)
        neigh_sets = {key: set(val) for key, val in neigh.items()}
        control_sets = {key: set(val) for key, val in control.items()}
        self.assertEqual(neigh_sets,control_sets,'the get_neighbour method failed to reproduce correct neighbour pairs for a single face of an fcc lattice')

    def test_get_neighbours_cross_lattice(self):
        
        control={0: [0, 2, 5, 6], 1: [1, 2, 5, 7], 2: [0, 1, 2, 3, 4, 5, 6, 7, 8], 3: [2, 3, 6, 8], 4: [2, 4, 7, 8]}
        sphere_centers_long, _,_=partition_cubic_volume(np.ones(3) * self.box_len,self.num_vol_all,self.sph_diam,flag='norand')

        sphere_centers_short, _,_=partition_cubic_volume(np.ones(3) * self.box_len,self.num_vol_side,self.sph_diam,flag='norand')

        neigh=get_neighbours_cross_lattice(sphere_centers_short,sphere_centers_long,np.ones(3) * self.box_len,cuttoff=self.sph_diam)
        self.assertEqual(neigh,control,'the get_neighbour method failed to reproduce correct neighbour pairs for a single face of an fcc lattice')  

    def test_get_neighbours_rectangular(self):
        box = self.rect_box
        cut = 2.0
        points = np.array(
            [
                # Across x-boundary (wrap), within cutoff.
                [0.5, 10.0, 15.0],
                [9.7, 10.0, 15.0],
                # Central point should not be within cutoff of either.
                [5.0, 10.0, 15.0],
            ]
        )
        neigh = get_neighbours(points, box, cuttoff=cut)

        self.assertIn(1, neigh[0])
        self.assertIn(0, neigh[1])
        self.assertNotIn(2, neigh[0])
        self.assertNotIn(2, neigh[1])

    def test_get_neighbours_cross_lattice_rectangular(self):
        box = self.rect_box
        cut = 2.0
        lattice_a = np.array(
            [
                # Nearest neighbor only across x-boundary.
                [0.5, 10.0, 15.0],
                # Nearest neighbor only inside the box.
                [5.0, 10.0, 15.0],
            ]
        )
        lattice_b = np.array(
            [
                [9.7, 10.0, 15.0],
                [5.9, 10.0, 15.0],
                # Far in y; should be excluded.
                [5.0, 18.5, 15.0],
            ]
        )
        neigh = get_neighbours_cross_lattice(lattice_a, lattice_b, box, cuttoff=cut)

        expected = {0: [0], 1: [1]}
        self.assertEqual(neigh, expected)

    def test_get_neighbours_multiple_rectangular(self):
        box = self.rect_box
        cut = 0.35
        points = np.array(
            [
                # Point at origin corner.
                [0.2, 0.2, 0.2],
                # Within cutoff across x-boundary.
                [9.9, 0.2, 0.2],
                # Within cutoff across y-boundary.
                [0.2, 19.9, 0.2],
                # Within cutoff across z-boundary.
                [0.2, 0.2, 29.9],
                # Far center point.
                [5.0, 10.0, 15.0],
            ]
        )
        neigh = get_neighbours(points, box, cuttoff=cut)
        expected = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
        neigh_sets = {key: set(val) for key, val in neigh.items()}
        expected_sets = {key: set(val) for key, val in expected.items()}
        self.assertEqual(neigh_sets, expected_sets)

    def test_get_neighbours_cross_lattice_multiple_rectangular(self):
        box = self.rect_box
        cut = 0.35
        lattice_a = np.array(
            [
                # Should match two neighbors across x/y boundaries.
                [0.2, 0.2, 0.2],
                # Should match only the nearby internal point.
                [5.0, 10.0, 15.0],
            ]
        )
        lattice_b = np.array(
            [
                [9.9, 0.2, 0.2],
                [0.2, 19.9, 0.2],
                [5.1, 10.0, 15.0],
                # Far away; should be excluded.
                [8.0, 8.0, 8.0],
            ]
        )
        neigh = get_neighbours_cross_lattice(lattice_a, lattice_b, box, cuttoff=cut)

        expected = {0: [0, 1], 1: [2]}
        self.assertEqual(neigh, expected)
        
# control={0: [0, 1, 2, 3, 5, 6, 9],
        #      1: [ 0,  1,  2,  4,  5,  7, 10],
        #      2: [ 0,  1,  2,  3,  4,  5,  6,  7,  8, 11],
        #      3: [ 0,  2,  3,  4,  6,  8, 12],
        #      4:[ 1,  2,  3,  4,  7,  8, 13]}

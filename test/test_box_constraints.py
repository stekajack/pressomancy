from create_system import sim_inst, BaseTestCase
import numpy as np
import espressomd


class BoxWallsTest(BaseTestCase):

    def setUp(self) -> None:
        self.box_len = sim_inst.sys.box_l

        self.sph_sigma_1=1
        self.sph_sigma_2=2
        self.sph_sigma_3=3

        sim_inst.sys.part.add(pos=(0,0,0), type=1)
        sim_inst.sys.part.add(pos=(0,0,0), type=2)
        sim_inst.sys.part.add(pos=(0,0,0), type=3)

        sim_inst.sys.non_bonded_inter[1,1].wca.set_params(epsilon=1, sigma=self.sph_sigma_1)
        sim_inst.sys.non_bonded_inter[2,2].wca.set_params(epsilon=1, sigma=self.sph_sigma_2)
        sim_inst.sys.non_bonded_inter[3,3].wca.set_params(epsilon=1, sigma=self.sph_sigma_3)

    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)
        self.assertListEqual(list(), list(sim_inst.sys.constraints))
        for typ in [1,2,3]:
            sim_inst.sys.non_bonded_inter[typ,typ].wca.deactivate()
            self.assertEqual(sim_inst.sys.non_bonded_inter[typ,typ].wca.sigma, 0)
            self.assertEqual(sim_inst.sys.non_bonded_inter[typ,typ].wca.epsilon, 0)

    def assertListOfTuplesAlmostEqual(self, expected, actual, message):
        self.assertEqual(len(expected), len(actual), f"{message}")
        for i, (exp_tuple, act_tuple) in enumerate(zip(expected, actual)):
            self.assertEqual(len(exp_tuple), len(act_tuple), f"{message}")
            for j, (exp_val, act_val) in enumerate(zip(exp_tuple, act_tuple)):
                if isinstance(exp_val, np.ndarray) or isinstance(act_val, np.ndarray):
                    # Ensure that the arrays are writable by copying them
                    np.testing.assert_allclose(np.copy(exp_val), np.copy(act_val),
                        err_msg=f"{message}")
                else:
                    self.assertEqual(exp_val, act_val,
                        f"{message}")

    def test_add_generic_box(self):
        control_walls=[]
        control_walls.append(espressomd.constraints.ShapeBasedConstraint(shape=espressomd.shapes.Wall(dist=0, normal=[0,0,1]))) #bottom
        control_walls.append(espressomd.constraints.ShapeBasedConstraint(shape=espressomd.shapes.Wall(dist=-self.box_len[2], normal=[0,0,-1]))) #top
        control_walls.append(espressomd.constraints.ShapeBasedConstraint(shape=espressomd.shapes.Wall(dist=0, normal=[0,1,0]))) #left
        control_walls.append(espressomd.constraints.ShapeBasedConstraint(shape=espressomd.shapes.Wall(dist=-self.box_len[1], normal=[0,-1,0]))) #right
        control_walls.append(espressomd.constraints.ShapeBasedConstraint(shape=espressomd.shapes.Wall(dist=0, normal=[1,0,0]))) #back
        control_walls.append(espressomd.constraints.ShapeBasedConstraint(shape=espressomd.shapes.Wall(dist=-self.box_len[0], normal=[-1,0,0]))) #front
        
        box_constraints = sim_inst.add_box_constraints()

        control_walls = [(wall.shape.dist, wall.shape.normal) for wall in control_walls]
        walls_inside = [(wall.shape.dist, wall.shape.normal) for wall in list(sim_inst.sys.constraints)]
        walls_return = [(wall.shape.dist, wall.shape.normal) for wall in box_constraints]

        self.assertListOfTuplesAlmostEqual(control_walls, walls_inside, "add_box_constraints does not create expected walls.")
        self.assertListOfTuplesAlmostEqual(walls_return, walls_inside, "add_box_constraints does not return the correct configuration.")

    def test_remove_generic_box(self):
        box_constraints = sim_inst.add_box_constraints()

        sim_inst.remove_box_constraints()

        self.assertListEqual(list(), list(sim_inst.sys.constraints), "remove_box_constraints did not fully remove walls.")

    def test_add_custom_box_with_wca(self):
        control_walls=[espressomd.constraints.ShapeBasedConstraint(shape=espressomd.shapes.Wall(dist=-self.box_len[2] / 2, normal=[0,0,-1])),]

        box_constraints = sim_inst.add_box_constraints(wall_type=0, sides='top', top=self.box_len[2]/2, inter='wca', types_=(1,2,3))

        control_walls = [(wall.shape.dist, wall.shape.normal) for wall in control_walls]
        walls_inside = [(wall.shape.dist, wall.shape.normal) for wall in list(sim_inst.sys.constraints)]
        walls_return = [(wall.shape.dist, wall.shape.normal) for wall in box_constraints]


        self.assertListOfTuplesAlmostEqual(control_walls, walls_inside, "add_box_constraints does not create expected walls.")
        self.assertListOfTuplesAlmostEqual(walls_return, walls_inside, "add_box_constraints does not return the correct configuration.")

        for typ in [1,2,3]:
            self.assertEqual(sim_inst.sys.non_bonded_inter[typ,0].wca.sigma, sim_inst.sys.non_bonded_inter[typ,typ].wca.sigma/2 / 2**(1/6), "wca sigma is not set up correctly")
            self.assertLess(0, sim_inst.sys.non_bonded_inter[typ,0].wca.epsilon, "wca was not set up, or was set up incorrectly")

    def test_remove_custom_box_with_wca(self):      
        box_constraints = sim_inst.add_box_constraints(wall_type=0, sides='no-sides', top=self.box_len[2]/2, inter='wca', types_=(1,2,3))

        wall_bottom = box_constraints[0]
        wall_top = box_constraints[1]

        #test 1
        sim_inst.remove_box_constraints(part_types=1)

        self.assertEqual(sim_inst.sys.non_bonded_inter[1,0].wca.sigma, 0, "wca not removed")
        self.assertEqual(0, sim_inst.sys.non_bonded_inter[1,0].wca.epsilon, "wca not removed")
        for typ in [2,3]:
            self.assertEqual(sim_inst.sys.non_bonded_inter[typ,0].wca.sigma, sim_inst.sys.non_bonded_inter[typ,typ].wca.sigma/2 / 2**(1/6), "wrong wca removed")
            self.assertLess(0, sim_inst.sys.non_bonded_inter[typ,0].wca.epsilon, "wrong wca removed")

        #test 2
        sim_inst.remove_box_constraints(wall_top)

        self.assertEqual(sim_inst.sys.non_bonded_inter[1,0].wca.sigma, 0, "wca recovered badly")
        self.assertEqual(0, sim_inst.sys.non_bonded_inter[1,0].wca.epsilon, "wca recovered badly")
        for typ in [2,3]:
            self.assertEqual(sim_inst.sys.non_bonded_inter[typ,0].wca.sigma, sim_inst.sys.non_bonded_inter[typ,typ].wca.sigma/2 / 2**(1/6), "wrong wca removed")
            self.assertLess(0, sim_inst.sys.non_bonded_inter[typ,0].wca.epsilon, "wrong wca removed")

        self.assertEqual(len(list(sim_inst.sys.constraints)), 1)
        self.assertEqual(list(sim_inst.sys.constraints)[0], wall_bottom)

        #test 3
        sim_inst.remove_box_constraints()

        self.assertListEqual(list(), list(sim_inst.sys.constraints), "remove_box_constraints did not fully remove walls.")

        for typ in [1,2,3]:
            self.assertEqual(sim_inst.sys.non_bonded_inter[typ,0].wca.sigma, 0, "wca not removed")
            self.assertEqual(0, sim_inst.sys.non_bonded_inter[typ,0].wca.epsilon, "wca not removed")
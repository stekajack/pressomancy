import numpy as np
import espressomd
from pressomancy.helper_functions import BondWrapper
from pressomancy.object_classes import Filament, Quartet, Quadriplex, RaspberrySphere
from create_system import sim_inst , BaseTestCase

class FilamentTest(BaseTestCase):

    part_per_fil = 4
    pos = np.array([[float(iid), 0., 0.] for iid in range(part_per_fil)])
    ori = np.tile(np.array([[1., 0., 0.]]), (part_per_fil, 1))

    def tearDown(self) -> None:
        self.instance=None
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part),0)

    def setUp(self) -> None:
        self.instance = Filament(config=Filament.config.specify(
            n_parts=self.part_per_fil, espresso_handle=sim_inst.sys))
        sim_inst.store_objects([self.instance])
        self.instance.set_object(pos=self.pos, ori=self.ori)

    def make_stuff(self):
        quartets = [Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys)) for _ in range(3*self.part_per_fil)]
        sim_inst.store_objects(quartets)
        quadriplexes = []
        for start in range(0, len(quartets), 3):
            grouped = quartets[start:start + 3]
            quadriplex = Quadriplex(config=Quadriplex.config.specify(
                associated_objects=grouped, espresso_handle=sim_inst.sys, bonding_mode='ftf'))
            quadriplexes.append(quadriplex)
        sim_inst.store_objects(quadriplexes)
        instance = Filament(config=Filament.config.specify(
            n_parts=self.part_per_fil, espresso_handle=sim_inst.sys, associated_objects=quadriplexes))
        sim_inst.store_objects([instance])
        pos = np.array([[0., 0., 6. * iid] for iid in range(self.part_per_fil)])
        ori = np.tile(np.array([[0., 0., 1.]]), (self.part_per_fil, 1))
        instance.set_object(pos=pos, ori=ori)
        return quadriplexes, instance

    def test_add_anchors(self):
        self.instance.add_anchors('real')
        self.assertEqual(len(self.instance.fronts_indices), self.part_per_fil)
        self.assertEqual(len(self.instance.backs_indices), self.part_per_fil)
        front_handles = list(sim_inst.sys.part.by_ids(self.instance.fronts_indices))
        back_handles = list(sim_inst.sys.part.by_ids(self.instance.backs_indices))
        self.assertEqual({handle.vs_relative[0] for handle in front_handles}, {part.id for part in self.instance.type_part_dict['real']})
        self.assertEqual({handle.vs_relative[0] for handle in back_handles}, {part.id for part in self.instance.type_part_dict['real']})

    def test_bond_anchors(self):
        self.instance.add_anchors('real')
        self.instance.bond_anchors()

        front_handles = list(sim_inst.sys.part.by_ids(self.instance.fronts_indices))
        self.assertEqual([bond[0][1] for bond in [handle.bonds for handle in front_handles[:-1]]], self.instance.backs_indices[1:])
        self.assertEqual(front_handles[-1].bonds, ())

    def test_bond_overlapping_virtualz(self):
        self.instance.add_anchors('real')
        self.instance.bond_overlapping_virtualz(crit=0.)
        front_handles = list(sim_inst.sys.part.by_ids(self.instance.fronts_indices))
        self.assertEqual([bond[0][1] for bond in [handle.bonds for handle in front_handles[:-1]]], self.instance.backs_indices[1:])
        self.assertEqual(front_handles[-1].bonds, ())

    def test_add_dipole_to_embedded_virt(self):
        self.instance.add_dipole_to_embedded_virt(type_name='real', dip_magnitude=2.)
        self.assertEqual(len(self.instance.magnetizable_virts), self.part_per_fil)
        virt_handles = list(sim_inst.sys.part.by_ids(self.instance.magnetizable_virts))
        self.assertEqual({handle.type for handle in virt_handles}, {self.instance.part_types['to_be_magnetized']})
        self.assertEqual({handle.vs_relative[0] for handle in virt_handles}, {part.id for part in self.instance.type_part_dict['real']})

    def test_add_dipole_to_type(self):
        self.instance.add_dipole_to_type('real', dip_magnitude=3.)
        for part in self.instance.type_part_dict['real']:
            self.assertTrue(np.allclose(part.dipm, 3))

    def test_bond_center_to_center(self):
        self.instance.bond_center_to_center(type_name='real')
        self.assertEqual([bond[0][1] for bond in [part.bonds for part in self.instance.type_part_dict['real'][:-1]]], [part.id for part in self.instance.type_part_dict['real'][1:]])
        self.cleanup()
        quartets = [Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys)) for _ in range(self.part_per_fil)]
        sim_inst.store_objects(quartets)
        self.instance = Filament(config=Filament.config.specify(
            n_parts=self.part_per_fil,espresso_handle=sim_inst.sys,associated_objects=quartets))
        sim_inst.store_objects([self.instance])
        pos = np.array([[0., 0., 6. * iid] for iid in range(self.part_per_fil)])
        ori = np.tile(np.array([[0., 0., 1.]]), (self.part_per_fil, 1))
        self.instance.set_object(pos=pos, ori=ori)
        self.instance.bond_center_to_center(type_name='real')
        self.assertEqual([quartet.type_part_dict['real'][0].bonds[0][1] for quartet in quartets[:-1]], [quartet.type_part_dict['real'][0].id for quartet in quartets[1:]])

    def test_bond_nearest_part(self):

        rasp_sigm = 3
        sample_bond_r0 = 0.83
        spacing = rasp_sigm + sample_bond_r0
        raspberry_equilibrium_r0 = 0.9197313953641069
        raspberries_config = RaspberrySphere.config.specify(
            sigma=1, size=rasp_sigm, espresso_handle=sim_inst.sys)
        raspberries = [RaspberrySphere(config=raspberries_config) for _ in range(self.part_per_fil)]
        sim_inst.store_objects(raspberries)
        bond_hndl = BondWrapper(espressomd.interactions.FeneBond(k=10, d_r_max=3 * rasp_sigm, r_0=raspberry_equilibrium_r0))
        size = self.part_per_fil * raspberries[0].params['size'] + sample_bond_r0 * (self.part_per_fil - 1)
        self.instance = Filament(config=Filament.config.specify(
            n_parts=self.part_per_fil,
            espresso_handle=sim_inst.sys,
            bond_handle=bond_hndl,
            associated_objects=raspberries,
        ))
        sim_inst.store_objects([self.instance])
        pos = np.array([[0., 0., spacing * idx] for idx in range(self.part_per_fil)])
        ori = np.tile(np.array([[0., 0., 1.]]), (self.part_per_fil, 1))
        self.instance.set_object(pos=pos, ori=ori)
        self.instance.bond_nearest_part('virt')
        sim_inst.sys.integrator.run(0)
        energy = sim_inst.sys.analysis.energy()

        no_bonds = sum(len(part.bonds) for raspberry in raspberries for part in raspberry.type_part_dict['virt'])
        self.assertEqual(no_bonds, self.part_per_fil - 1)
        self.assertAlmostEqual(energy['bonded'], 0)

    def test_bending_potential(self):
        angle_harmonic = espressomd.interactions.AngleHarmonic(bend=1., phi0=3.)
        self.instance.add_bending_potential(type_name='real',bond_handle=angle_harmonic)
        self.assertEqual(self.instance.type_part_dict['real'][0].bonds, ())
        self.assertEqual(self.instance.type_part_dict['real'][-1].bonds, ())
        for iid in range(1, self.part_per_fil - 1):
            self.assertEqual(self.instance.type_part_dict['real'][iid].bonds[0][0], angle_harmonic)
            self.assertEqual(
                self.instance.type_part_dict['real'][iid].bonds[0][1:],
                (self.instance.type_part_dict['real'][iid + 1].id, self.instance.type_part_dict['real'][iid - 1].id),
            )

    def test_bond_quadriplexes(self):
        self.cleanup()
        quadriplexes,instance=self.make_stuff()
        pre_hinge_bonds = sum(len(corner.bonds) for quadriplex in quadriplexes for quartet in quadriplex.associated_objects for corner in quartet.corner_particles)
        instance.bond_quadriplexes(mode='hinge')
        post_hinge_bonds = sum(len(corner.bonds) for quadriplex in quadriplexes for quartet in quadriplex.associated_objects for corner in quartet.corner_particles)
        self.assertEqual(post_hinge_bonds - pre_hinge_bonds, self.part_per_fil - 1)
        sim_inst.reinitialize_instance()
        quadriplexes,instance=self.make_stuff()
        pre_all_bonds = sum(len(corner.bonds) for quadriplex in quadriplexes for quartet in quadriplex.associated_objects for corner in quartet.corner_particles)
        instance.bond_quadriplexes(mode='all')
        post_all_bonds = sum(len(corner.bonds) for quadriplex in quadriplexes for quartet in quadriplex.associated_objects for corner in quartet.corner_particles)
        self.assertEqual(post_all_bonds - pre_all_bonds, 4 * (self.part_per_fil - 1))

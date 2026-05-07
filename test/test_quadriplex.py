import numpy as np
import pressomancy.object_classes
import espressomd
from create_system import sim_inst , BaseTestCase
from pressomancy.object_classes.quadriplex_class import *


class QuadriplexTest(BaseTestCase):
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam

    def setUp(self) -> None:
        self.instance_ftf=Quadriplex(config=Quadriplex.config.specify(espresso_handle=sim_inst.sys, bonding_mode='ftf'))
        self.instance_ctc=Quadriplex(config=Quadriplex.config.specify(espresso_handle=sim_inst.sys, bonding_mode='ctc'))
        sim_inst.store_objects([self.instance_ftf, self.instance_ctc])

    def tearDown(self) -> None:
        self.instance_ftf=None
        self.instance_ctc=None
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part),0)

    def check_bond_energy(self,bond_hndl, tag):
        sim_inst.sys.integrator.run(0)
        energy = sim_inst.sys.analysis.energy()
        no_bonds=0
        for keys,val in energy.items():
            if keys=='bonded':
                no_bonds+=1
                self.assertAlmostEqual(val, 0, msg=f"unexpected energy after adding {bond_hndl.__class__.__name__} to {tag}: expected=0, got={val}")
        self.assertEqual(no_bonds, 1, msg=f"unexpected number of bonds after adding {bond_hndl.__class__.__name__} to {tag}: expected=1, got={no_bonds}")

    @staticmethod
    def make_obj(quartet_types):
        quartet_triplet = []
        for quartet_type in quartet_types:
            quartet_config = Quartet.config.specify(
                alias='quartet',
                type=quartet_type,
                espresso_handle=sim_inst.sys,
            )
            quartet = Quartet(config=quartet_config)
            quartet_triplet.append(quartet)
        quadriplex_config = Quadriplex.config.specify(
            associated_objects=quartet_triplet,
            espresso_handle=sim_inst.sys,
            bonding_mode='ftf',
            size=np.sqrt(3)*5,
        )
        quadriplex=Quadriplex(config=quadriplex_config)
        sim_inst.store_objects([quadriplex,])
        quadriplex.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        return quartet_triplet,quadriplex

    def test_add_patches_triples(self):
        self.instance_ftf.set_object(
            pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance_ftf.add_patches_triples()
        patch_parts = self.instance_ftf.type_part_dict['patch']
        self.assertEqual(len(patch_parts), 2)
        top_real = self.instance_ftf.associated_objects[1].type_part_dict['real'][0]
        bottom_real = self.instance_ftf.associated_objects[2].type_part_dict['real'][0]
        owner_ids = {top_real.id, bottom_real.id}
        self.assertEqual({patch.vs_relative[0] for patch in patch_parts}, owner_ids)

        patch_by_owner = {patch.vs_relative[0]: patch for patch in patch_parts}
        self.assertEqual(set(int(part_id) for part_id in patch_by_owner[top_real.id].exclusions), {patch_by_owner[bottom_real.id].id})
        self.assertEqual(set(int(part_id) for part_id in patch_by_owner[bottom_real.id].exclusions), {patch_by_owner[top_real.id].id})

    def test_add_bending_potential(self):

        def asserts():
            for central, top, bottom in zip(center_parts, top_parts, bottom_parts):
                angle_bonds = [bond for bond in central.bonds if bond[0] == int_nhdl]
                self.assertEqual(len(angle_bonds), 1)
                self.assertEqual(angle_bonds[0][1:], (top.id, bottom.id))
                self.assertEqual(sim_inst.sys.part.by_id(central.id).bonds, central.bonds)

        sim_inst.sys.integrator.run(0)
        energy = sim_inst.sys.analysis.energy()
        self.assertAlmostEqual(energy['bonded'], 0, msg=f"unexpected bonded energy before adding bending potential: expected=0, got={energy['bonded']}")
        int_nhdl=espressomd.interactions.AngleHarmonic(bend=1, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(int_nhdl)
        self.instance_ftf.set_object(
            pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance_ftf.add_bending_potential(bending_potential_handle=int_nhdl)
        self.check_bond_energy(int_nhdl,'ftf quadriplex')
        center_parts = self.instance_ftf.associated_objects[0].corner_particles
        top_parts = self.instance_ftf.associated_objects[1].corner_particles
        bottom_parts = self.instance_ftf.associated_objects[2].corner_particles
        asserts()
        self.instance_ctc.set_object(
            pos=np.array([10,10,10]),ori=np.array([0,0,1]))
        self.instance_ctc.add_bending_potential(bending_potential_handle=int_nhdl)
        self.check_bond_energy(int_nhdl, 'ctc quariplex')
        center_parts = np.atleast_1d(self.instance_ctc.associated_objects[0].type_part_dict['real'][0])
        top_parts = np.atleast_1d(self.instance_ctc.associated_objects[1].type_part_dict['real'][0])
        bottom_parts = np.atleast_1d(self.instance_ctc.associated_objects[2].type_part_dict['real'][0])
        asserts()

    def test_add_dihedrals(self):
        fold_dict={ 'antiparallel': ['brokenB', 'brokenA', 'brokenA'],'hybrid':['brokenA', 'brokenB', 'brokenA'],'pareallel':['brokenA', 'brokenA', 'brokenA']}

        for fold,quartet_types in fold_dict.items():
            quartet_triplet,quadriplex=self.make_obj(quartet_types)
            for quartet in quartet_triplet:
                quartet.add_h_bond_patches()
            sim_inst.sys.integrator.run(0)
            energy = sim_inst.sys.analysis.energy()
            self.assertAlmostEqual(energy['bonded'], 0, msg=f"unexpected bonded energy before adding bending potential: expected=0, got={energy['bonded']}")
            dihedral = espressomd.interactions.Dihedral(bend=10, mult=1, phase=np.pi/2.)
            sim_inst.sys.bonded_inter.add(dihedral)
            quadriplex.add_dihedrals(dihedral_potential_handle=dihedral)
            self.check_bond_energy(dihedral, f'{fold} quadripex')
            self.cleanup()

    def test_add_extra_bendings(self):
        fold_dict={ 'antiparallel': ['brokenB', 'brokenA', 'brokenA'],'hybrid':['brokenA', 'brokenB', 'brokenA'],'pareallel':['brokenA', 'brokenA', 'brokenA']}

        for fold,quartet_types in fold_dict.items():
            quartet_triplet,quadriplex=self.make_obj(quartet_types)
            for quartet in quartet_triplet:
                quartet.add_h_bond_patches()
            sim_inst.sys.integrator.run(0)
            energy = sim_inst.sys.analysis.energy()
            self.assertAlmostEqual(energy['bonded'], 0, msg=f"unexpected bonded energy before adding bending potential: expected=0, got={energy['bonded']}")
            angle_another = espressomd.interactions.AngleHarmonic(bend=10.0, phi0=np.pi/2.)
            sim_inst.sys.bonded_inter.add(angle_another)
            quadriplex.add_extra_bendings(bending_potential_handle=angle_another)
            self.check_bond_energy(angle_another, f'{fold} quadripex')
            self.cleanup()

    def test_mark_covalent_bonds(self):
        self.instance_ftf.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance_ftf.mark_covalent_bonds(part_type=666)

        top_quartet = self.instance_ftf.associated_objects[1]
        bottom_quartet = self.instance_ftf.associated_objects[2]
        self.assertEqual(sum(corner.type == 666 for corner in top_quartet.corner_particles), 1)
        self.assertEqual(sum(corner.type == 666 for corner in bottom_quartet.corner_particles), 1)

class QuartetTest(BaseTestCase):

    def tearDown(self) -> None:
        self.cleanup()
        self.assertEqual(len(sim_inst.objects),0)

    def test_add_h_bond_patches(self):
        for quartet_alias,quartet_type in product(['quartet', 'quartet_11x11'],['brokenA', 'brokenB']):
            with self.subTest(alias=quartet_alias, quartet_type=quartet_type):
                quartet = Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys, type=quartet_type, alias=quartet_alias))
                sim_inst.store_objects([quartet])
                quartet.set_object(pos=np.array([0, 0, 0]), ori=np.array([0, 0, 1]))
                quartet.add_h_bond_patches()
                parts, _ = quartet.get_owned_part()
                square_a_type = quartet.part_types['squareA']
                square_b_type = quartet.part_types['squareB']

                patch_map = {}
                for corner in quartet.corner_particles:
                    related = [part for part in parts if part.vs_relative[0] == corner.id]
                    square_a_parts = [part for part in related if part.type == square_a_type]
                    square_b_parts = [part for part in related if part.type == square_b_type]
                    self.assertEqual(len(square_a_parts), 2)
                    self.assertEqual(len(square_b_parts), 2)
                    patch_map[corner.id] = {'squareA': square_a_parts, 'squareB': square_b_parts}

                for left_corner in quartet.corner_particles:
                    partner_count = 0
                    for right_corner in quartet.corner_particles:
                        if right_corner.id == left_corner.id:
                            continue
                        overlaps = sum(
                            np.isclose(
                                np.linalg.norm(np.array(left_patch.pos) - np.array(right_patch.pos)),
                                0.0,
                                atol=1e-8,
                            )
                            for left_patch in patch_map[left_corner.id]['squareB']
                            for right_patch in patch_map[right_corner.id]['squareA']
                        )
                        if overlaps == 2:
                            partner_count += 1
                    self.assertEqual(
                        partner_count,
                        1,
                        msg=f"Patch partner mismatch alias={quartet_alias} type={quartet_type} corner={left_corner.id}",
                    )
                self.cleanup()

    def test_mark_covalent_corners(self):
        quartet = Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys, type='brokenA', alias='quartet'))
        sim_inst.store_objects([quartet])
        quartet.set_object(pos=np.array([0, 0, 0]), ori=np.array([0, 0, 1]))
        quartet.mark_covalent_corner(part_type=666)

        self.assertEqual(sum(corner.type == 666 for corner in quartet.corner_particles), 1)

    def test_object_contracts(self):
        for quartet_alias, (quartet_type, pos) in product(['quartet', 'quartet_11x11'], zip(['solid', 'brokenA', 'brokenB'], [(0, 0, 0), (10, 10, 10), (20, 20, 20)])):
            with self.subTest(alias=quartet_alias, quartet_type=quartet_type):
                quartet = Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys, type=quartet_type, alias=quartet_alias))
                sim_inst.store_objects([quartet])
                quartet.set_object(pos=pos, ori=np.array([0, 0, 1]))
                parts, _ = quartet.get_owned_part()
                tracked_ids = set()

                if quartet.params['type'] == 'solid':
                    expected_no_excl = len(parts) - 1
                else:
                    recipe = quartet.recepie_dictA if quartet.params['type'] == 'brokenA' else quartet.recepie_dictB
                    expected_no_excl = len(next(iter(recipe['assoc'].values())))

                for type_name, expected_type in quartet.part_types.items():
                    self.assertIn(type_name, sim_inst.part_types)
                    self.assertEqual(sim_inst.part_types[type_name], expected_type)
                    for part in quartet.type_part_dict[type_name]:
                        tracked_ids.add(part.id)
                        self.assertEqual(part.type, expected_type)
                        core_part = sim_inst.sys.part.by_id(part.id)
                        self.assertEqual(core_part.type, expected_type)
                        expected_exclusions = 0 if type_name == 'cation' else expected_no_excl
                        self.assertEqual(
                            len(core_part.exclusions),
                            expected_exclusions,
                            msg=f"unexpected number of exclusions for alias={quartet.params['alias']} type={quartet.params['type']} particle_type={type_name}: expected={expected_exclusions}, got={len(core_part.exclusions)}",
                        )

                self.assertEqual(tracked_ids, {part.id for part in parts})
                self.cleanup()

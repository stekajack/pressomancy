from itertools import combinations, product
import random
import warnings
import logging
import espressomd
import numpy as np
from pressomancy.helper_functions import PartDictSafe, SinglePairDict, align_vectors
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.object_classes.rigid_obj import GenericRigidObj
from pressomancy.helper_functions import BondWrapper

class Quartet(GenericRigidObj):

    '''
    Class that contains quartet relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quartet. Therefore many relevant parameters are class specific, not instance specific.
    '''
    numInstances = 0
    
    recepie_dictA = {'assoc': {1: [2, 3, 6, 7, 8], 
                               5: [4, 9, 10, 13, 14], 
                               20: [11, 12, 15, 16, 21], 
                               24: [17, 18, 19, 22, 23]}, 
                    'circ': [8, 13, 12, 17], 
                    'squareA': [6, 4, 21, 19], 
                    'squareB': [11, 3, 22, 14], 
                    'squareC': [7, 9, 16, 18]}
    
    recepie_dictB = {'assoc': {1: [2, 6, 7, 11, 12], 
                               5: [4, 9, 10, 8, 3], 
                               20: [15, 16, 21, 22, 17], 
                               24: [18, 19, 23, 13, 14]}, 
                    'circ': [8, 13, 12, 17], 
                    'squareA': [2, 10, 15, 23], 
                    'squareB': [11, 3,  22, 14], 
                    'squareC': [7, 9, 16, 18]}
    part_types = PartDictSafe()
    config = ObjectConfigParams(
        n_parts=25,
        alias='quartet',
        type='solid', 
        bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0))
    )

    def __init__(self,config: ObjectConfigParams):
        '''
        Initialisation of a quartet object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        super().__init__(config)
        assert config['type'] in ['solid', 'brokenA', 'brokenB'], 'type must be either solid, brokenA or brokenB!!!'
        assert config['n_parts'] ==len(self._reference_sheet[self.params['alias']]), 'n_parts must be equal to the number of parts in the reference sheet!!!'
        if self.params['type'] in ['brokenA', 'brokenB']:
            assert self.params['bond_handle'] != None, 'broken quartets require a bond to be set!!!'
            Quartet.part_types.update({'circ': 28,
                  'squareA': 24, 'squareB': 25, 'cation': 27})
        self.who_am_i = Quartet.numInstances
        Quartet.numInstances += 1
        self.orientor = np.empty(shape=3, dtype=float)
        self.corner_particles = []

    def set_object(self,  pos, ori, triplet=None):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part. Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        rotation_matrix = align_vectors(np.array([0.0,0.0,1.0]),ori) # 0,0,1 is the default director in espressomd
        positions = np.dot(self._reference_sheet[self.params['alias']],rotation_matrix.T) + pos 
        particles=[self.add_particle(type_name='virt',pos=pos) for pos in positions]
        self.unperturbed_particles=particles
        diag = np.sqrt(2)*4
        self.corner_particles = [part for part0, part in product(
            particles, particles) if np.isclose(np.linalg.norm(part0.pos - part.pos), diag, atol=1e-06)]
        if self.params['type'] == 'solid':
            self.change_part_type(particles[0],'real')
            w, d, h = 4, 4, 0
            rot_inertia_x = 1/12*(h**2+d**2)
            rot_inertia_y = 1/12*(w**2+h**2)
            rot_inertia_z = 1/12*(w**2+d**2)

            particles[0].rotation = (True, True, True)
            particles[0].rinertia = (
                rot_inertia_x, rot_inertia_y, rot_inertia_z)
            particles[0].director = ori
            np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
                particles[0], particles[1:])
            for ii, jj in combinations(particles, 2):
                ii.add_exclusion(jj)

        if self.params['type'] == 'brokenA':
            self.change_part_type(particles[0],'cation')
            particles[0].q = 1

            
            particles = np.array(particles)
            for part in self.corner_particles:
                self.change_part_type(part,'real')
                part.rotation = (True, True, True)
                part.director = ori

            for part in particles[np.array(self.recepie_dictA['circ'])]:
                self.change_part_type(part,'circ')
                part.q = -0.25

            if self.params['bond_handle'] != None:
                for part1, part2 in zip(particles[np.array(self.recepie_dictA['squareA'])], particles[np.array(self.recepie_dictA['squareB'])]):
                    pass
                    # self.change_part_type(part1,'squareA')
                    # self.change_part_type(part2,'squareB')
            else:
                for part in particles[np.array(self.recepie_dictA['squareA'])]:
                    self.change_part_type(part,'squareA')

                for part in particles[np.array(self.recepie_dictA['squareB'])]:
                    self.change_part_type(part,'squareB')

            for key, values in self.recepie_dictA['assoc'].items():
                np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
                    particles[key], particles[values])
                for ii, jj in combinations([particles[key],]+[x for x in particles[values]], 2):
                    ii.add_exclusion(jj)

        if self.params['type'] == 'brokenB':
            self.change_part_type(particles[0],'cation')
            particles[0].q = 1

            particles = np.array(particles)
            for part in self.corner_particles:
                self.change_part_type(part,'real')
                part.rotation = (True, True, True)
                part.director = ori

            for part in particles[np.array(self.recepie_dictB['circ'])]:
                self.change_part_type(part,'circ')
                part.q = -0.25

            if self.params['bond_handle'] != None:
                for part1, part2 in zip(particles[np.array(self.recepie_dictB['squareA'])], particles[np.array(self.recepie_dictB['squareB'])]):
                    pass
                    # self.change_part_type(part1,'squareA')
                    # self.change_part_type(part2,'squareB')
            else:
                for part in particles[np.array(self.recepie_dictB['squareA'])]:
                    self.change_part_type(part,'squareA')

                for part in particles[np.array(self.recepie_dictB['squareB'])]:
                    self.change_part_type(part,'squareB')

            for key, values in self.recepie_dictB['assoc'].items():
                np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
                    particles[key], particles[values])
                for ii, jj in combinations([particles[key],]+[x for x in particles[values]], 2):
                    ii.add_exclusion(jj)

        return self

    def mark_covalent_corner(self, part_type=666):
        random_part = random.choice(self.corner_particles)
        random_part.type = part_type
        logging.info(f'covalent corners marked for {self.__class__.__name__}s with part_type {part_type}')
    
    def add_h_bond_patches(self):
        recepie_dict = self.recepie_dictA if self.params['type'] == 'brokenA' else self.recepie_dictB
        square_b_set = recepie_dict['squareB']
        square_a_set = recepie_dict['squareA']

        cation_offset = 0.5
        cation_radius=0.2

        # For each corner, place its associated cation just beyond the squareB site
        # along the corner->squareB direction.
        particles=self.unperturbed_particles
        for corner_idx, assoc_indices in recepie_dict['assoc'].items():
            square_b_matches = [idx for idx in assoc_indices if idx in square_b_set]
            square_a_matches = [idx for idx in assoc_indices if idx in square_a_set]
            if len(square_b_matches) != 1 or len(square_a_matches) != 1:
                raise ValueError(f'Expected exactly one squareB and one squareA match for corner index {corner_idx}, but found {len(square_b_matches)} squareB matches and {len(square_a_matches)} squareA matches')
            square_b_idx = square_b_matches[0]
            square_a_idx = square_a_matches[0]
            corner_part = particles[corner_idx]
            square_b_part = particles[square_b_idx]
            square_a_part = particles[square_a_idx]
            
            direction_b = corner_part.pos-square_b_part.pos
            direction_a = corner_part.pos-square_a_part.pos

            direction_norm_b = np.linalg.norm(direction_b)
            direction_norm_a = np.linalg.norm(direction_a)
            unit_direction_b = direction_b / direction_norm_b
            unit_direction_a = direction_a / direction_norm_a
            part_hndlB=self.add_particle(type_name='squareB',pos=square_b_part.pos - cation_offset * unit_direction_b)
            part_hndlB.vs_auto_relate_to(corner_part)
            part_hndlA=self.add_particle(type_name='squareA',pos=square_a_part.pos - cation_offset * unit_direction_a)
            part_hndlA.vs_auto_relate_to(corner_part)

            part_hndlBB=self.add_particle(type_name='squareB',pos=part_hndlB.pos  - direction_a)
            part_hndlBB.vs_auto_relate_to(corner_part)
            part_hndlAA=self.add_particle(type_name='squareA',pos=part_hndlA.pos - direction_b/2.)
            part_hndlAA.vs_auto_relate_to(corner_part)

            part_hndlA.add_exclusion(part_hndlB.id)
            part_hndlAA.add_exclusion(part_hndlBB.id)
            part_hndlAA.add_exclusion(part_hndlB.id)
            part_hndlBB.add_exclusion(part_hndlA.id)
            corner_part.pos = corner_part.pos + unit_direction_a*cation_radius + unit_direction_b*cation_radius


class Quadriplex(metaclass=Simulation_Object):

    '''
    Class that contains quadriplex relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    part_types = PartDictSafe()
    simulation_type=SinglePairDict('quadriplex',22)
    config = ObjectConfigParams(
        n_parts=3,
        size=6.,
        bonding_mode='ftf',
        bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a quadriplex object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        
        self.sys=config['espresso_handle']
        self.params=config
        if self.params['associated_objects']==None:
            warnings.warn('no associated_objects have been passed explicity. Creating objects required to initialise object implicitly!')
            configuration=Quartet.config.specify(espresso_handle=self.sys)
            self.params['associated_objects']=[Quartet(config=configuration) for _ in range(3)]
        self.associated_objects=self.params['associated_objects']
        assert config['n_parts'] == len(config['associated_objects']), f'n_parts must be equal to the number of associated objects!!! {config["n_parts"], len(config["associated_objects"])}'
        self.has_been_set=False
        self.who_am_i = Quadriplex.numInstances
        Quadriplex.numInstances += 1
        self.orientor = np.empty(shape=3, dtype=float)
        self.type_part_dict=PartDictSafe({key: [] for key in Quadriplex.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        if self.has_been_set:
            raise RuntimeError(f'object {self.__class__.__name__} wiht id {self.who_am_i} was attempted to be set but it already exists!!!')
        assert self.params['n_parts'] == 3, "a quadriplex can only be created from 3 quartets!!! "
        assert all([x.simulation_type==self.associated_objects[0].simulation_type for x in self.associated_objects[1:]]), 'all objects must have the same simulation type!'
        type_str=self.associated_objects[0].simulation_type.key

        p_central = self.associated_objects[0].set_object(
            pos, ori, triplet=self.who_am_i)
        p_top = self.associated_objects[1].set_object(pos+self.params['bond_handle'].r_0*ori,
                                                       ori, triplet=self.who_am_i)
        p_bottom = self.associated_objects[2].set_object(pos-self.params['bond_handle'].r_0*ori,
                                                          ori, triplet=self.who_am_i)
        if self.params['bonding_mode'] == 'ctc':
            self.bond_quartets_center_to_center()
        if self.params['bonding_mode'] == 'ftf':
            self.bond_quartets_corner_to_corner()
        self.has_been_set=True
        return self
         
    def add_patches_triples(self):

        self.part_types['patch'] = 4
        triples = self.associated_objects
        part_hndl_a=self.add_particle(type_name='patch', pos=triples[1].type_part_dict['real'][0].pos, director=triples[1].type_part_dict['real'][0].director)
        part_hndl_a.vs_auto_relate_to(triples[1].type_part_dict['real'][0])

        part_hndl_b=self.add_particle(type_name='patch', pos=triples[2].type_part_dict['real'][0].pos, director=triples[2].type_part_dict['real'][0].director)
        part_hndl_b.vs_auto_relate_to(triples[2].type_part_dict['real'][0])
        part_hndl_a.add_exclusion(part_hndl_b.id)        
 
    def bond_quartets_center_to_center(self):
        assert len(
            self.associated_objects) == 3, "a quadriplex can only be created from 3 quartets!!! "
        assert self.params['bonding_mode'] == 'ctc', 'this method is only valid for center to center bonding!!!'
        self.bond_owned_part_pair(self.associated_objects[0].type_part_dict['real'][0], self.associated_objects[1].type_part_dict['real'][0])   

        self.bond_owned_part_pair(self.associated_objects[0].type_part_dict['real'][0], self.associated_objects[2].type_part_dict['real'][0])

    def bond_quartets_corner_to_corner(self):
        assert len(
            self.associated_objects) == 3, "a quadriplex can only be created from 3 quartets!!! "
        assert self.params['bonding_mode'] == 'ftf', 'this method is only valid for corner to corner bonding!!!'
        candidate1, candidate2, candidate3, pair_distances = self.associated_objects[
            0].corner_particles, self.associated_objects[1].corner_particles, self.associated_objects[2].corner_particles, []

        pos1 = np.array([x.pos for x in candidate1])
        pos2 = np.array([x.pos for x in candidate2])
        pos3 = np.array([x.pos for x in candidate3])

        indices_int = np.arange(len(candidate1))
        index_combinations = np.array(
            list(product(indices_int, indices_int)))
        pair_distances = np.linalg.norm(
            pos1[index_combinations[:, 0]] - pos2[index_combinations[:, 1]], axis=-1)
        filtered_indices = index_combinations[np.isclose(
            pair_distances, self.params['bond_handle'].r_0)]
        for index_pair in filtered_indices:
            p1_id, p2_id = index_pair
            self.bond_owned_part_pair(candidate1[p1_id], candidate2[p2_id])

        pair_distances = np.linalg.norm(
            pos1[index_combinations[:, 0]] - pos3[index_combinations[:, 1]], axis=-1)
        filtered_indices = index_combinations[np.isclose(
            pair_distances, self.params['bond_handle'].r_0)]
        for index_pair in filtered_indices:
            p1_id, p2_id = index_pair
            self.bond_owned_part_pair(candidate1[p1_id], candidate3[p2_id])

    def add_bending_potential(self, bending_potential_handle):
        assert len(
            self.associated_objects) == 3, "a quadriplex can only be created from 3 quartets!!! "
        self.bending_potential_handle=bending_potential_handle
        logging.info(f'adding bending potential to {self.__class__.__name__} with id {self.who_am_i}')
        if self.params['bonding_mode'] == 'ctc':
            central = self.associated_objects[0].type_part_dict['real'][0]
            top = self.associated_objects[1].type_part_dict['real'][0]
            bottom = self.associated_objects[2].type_part_dict['real'][0]
            central.add_bond((self.bending_potential_handle, top,  bottom))

        if self.params['bonding_mode'] == 'ftf':
            center_corners = self.associated_objects[0].corner_particles
            top_corners = self.associated_objects[1].corner_particles
            bottom_corners = self.associated_objects[-1].corner_particles
            for central, top, bottom in zip(center_corners, top_corners, bottom_corners):
                central.add_bond(
                    (self.bending_potential_handle, top,  bottom))

    def mark_covalent_bonds(self, part_type=666):
        assert len(
            self.associated_objects) == 3, "a quadriplex can only be created from 3 quartets!!! "
        self.associated_objects[1].mark_covalent_corner(
            part_type=part_type)
        self.associated_objects[2].mark_covalent_corner(
            part_type=part_type)

    def _signed_dihedral(self, p0, p1, p2, p3):
        x0 = np.array(p0.pos)
        x1 = np.array(p1.pos)
        x2 = np.array(p2.pos)
        x3 = np.array(p3.pos)

        b0 = x0 - x1
        b1 = x2 - x1
        b2 = x3 - x2

        b1_norm = np.linalg.norm(b1)
        if np.isclose(b1_norm, 0.0):
            return 0.0
        b1_unit = b1 / b1_norm

        v = b0 - np.dot(b0, b1_unit) * b1_unit
        w = b2 - np.dot(b2, b1_unit) * b1_unit

        x = np.dot(v, w)
        y = np.dot(np.cross(b1_unit, v), w)
        # ESPResSo's dihedral sign convention is opposite to this raw geometric form.
        return -np.arctan2(y, x)

    def _angle_error(self, phi, target):
        delta = phi - target
        return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))

    def _build_corner_patch_map(self, quartet):
        parts, _ = quartet.get_owned_part()
        square_a_type = quartet.part_types['squareA']
        square_b_type = quartet.part_types['squareB']
        corner_to_patches = {}

        for corner in quartet.corner_particles:
            related_parts = [
                part for part in parts
                if part.vs_relative[0] == corner.id
            ]
            square_a_parts = [part for part in related_parts if part.type == square_a_type]
            square_b_parts = [part for part in related_parts if part.type == square_b_type]
            if len(square_a_parts) < 1 or len(square_b_parts) < 1:
                raise ValueError(
                    f'Invalid patch map for quartet_id={quartet.who_am_i} type={quartet.params["type"]} corner_id={corner.id}: '
                    f'squareA={len(square_a_parts)} squareB={len(square_b_parts)}'
                )
            square_a_parts_sorted = sorted(
                square_a_parts,
                key=lambda part: (np.linalg.norm(np.array(part.pos) - np.array(corner.pos)), part.id),
            )
            square_b_parts_sorted = sorted(
                square_b_parts,
                key=lambda part: (np.linalg.norm(np.array(part.pos) - np.array(corner.pos)), part.id),
            )
            corner_to_patches[corner.id] = {
                'squareA': square_a_parts_sorted[0],
                'squareB': square_b_parts_sorted[0],
            }
        return corner_to_patches

    def _nearest_corner(self, ref_corner, candidate_corners):
        pair_distances = np.array([np.linalg.norm(ref_corner.pos - corner.pos) for corner in candidate_corners])
        min_distance = np.min(pair_distances)
        closest_indices = np.flatnonzero(np.isclose(pair_distances, min_distance, atol=1e-8))
        if len(closest_indices) > 1:
            tied_corners = [candidate_corners[idx] for idx in closest_indices]
            return min(tied_corners, key=lambda part: part.id)
        return candidate_corners[closest_indices[0]]

    def _add_dihedrals_between(self, q_src, q_dst, dihedral_handle, target_phase=np.pi/2.):
        src_patch_map = self._build_corner_patch_map(q_src)
        dst_patch_map = self._build_corner_patch_map(q_dst)
        dst_corners = q_dst.corner_particles
        tie_break_order = [
            ('squareA', 'squareA', 'forward'),
            ('squareB', 'squareB', 'forward'),
            ('squareA', 'squareB', 'forward'),
            ('squareB', 'squareA', 'forward'),
            ('squareA', 'squareA', 'reverse'),
            ('squareB', 'squareB', 'reverse'),
            ('squareA', 'squareB', 'reverse'),
            ('squareB', 'squareA', 'reverse'),
        ]

        for src_corner in q_src.corner_particles:
            pair_distances = np.array([np.linalg.norm(src_corner.pos - corner.pos) for corner in dst_corners])
            min_distance = np.min(pair_distances)
            closest_indices = np.flatnonzero(np.isclose(pair_distances, min_distance, atol=1e-8))
            if len(closest_indices) > 1:
                tied_corners = [dst_corners[idx] for idx in closest_indices]
                dst_corner = min(tied_corners, key=lambda part: part.id)
                logging.debug(
                    'dihedral tie break: src_corner=%s dst_candidates=%s selected=%s',
                    src_corner.id,
                    [part.id for part in tied_corners],
                    dst_corner.id,
                )
            else:
                dst_corner = dst_corners[closest_indices[0]]

            src_patches = src_patch_map[src_corner.id]
            dst_patches = dst_patch_map[dst_corner.id]
            best_choice = None
            best_error = np.inf
            for src_key, dst_key, direction in tie_break_order:
                src_patch = src_patches[src_key]
                dst_patch = dst_patches[dst_key]
                if direction == 'forward':
                    phi = self._signed_dihedral(src_patch, src_corner, dst_corner, dst_patch)
                else:
                    phi = self._signed_dihedral(dst_patch, dst_corner, src_corner, src_patch)
                error = self._angle_error(phi, target_phase)
                if error < best_error - 1e-12:
                    best_error = error
                    best_choice = (src_key, dst_key, direction)

            if best_choice is None:
                raise RuntimeError(
                    f'No valid dihedral candidate for src_corner={src_corner.id}, dst_corner={dst_corner.id}'
                )
            src_key, dst_key, direction = best_choice
            if direction == 'forward':
                src_patches[src_key].add_bond((dihedral_handle, src_corner, dst_corner, dst_patches[dst_key]))
            else:
                dst_patches[dst_key].add_bond((dihedral_handle, dst_corner, src_corner, src_patches[src_key]))
        
    def add_dihedrals(self):
        assert len(
            self.associated_objects) == 3, "a quadriplex can only be created from 3 quartets!!! "
        center_quartet = self.associated_objects[0]
        top_quartet = self.associated_objects[1]
        bottom_quartet = self.associated_objects[2]

        dihedral = espressomd.interactions.Dihedral(bend=10, mult=1, phase=np.pi/2.)
        self.sys.bonded_inter.add(dihedral)
        self._add_dihedrals_between(top_quartet, center_quartet, dihedral, target_phase=np.pi/2.)
        self._add_dihedrals_between(center_quartet, bottom_quartet, dihedral, target_phase=np.pi/2.)

    def add_extra_bendings(self):
        angle_another = espressomd.interactions.AngleHarmonic(bend=10.0, phi0=np.pi/2.)
        self.sys.bonded_inter.add(angle_another)
        center_quartet = self.associated_objects[0]
        top_quartet = self.associated_objects[1]
        bottom_quartet = self.associated_objects[2]

        top_patch_map = self._build_corner_patch_map(top_quartet)
        center_patch_map = self._build_corner_patch_map(center_quartet)
        bottom_patch_map = self._build_corner_patch_map(bottom_quartet)

        for ref_corner in top_quartet.corner_particles:
            closest_corner = self._nearest_corner(ref_corner, center_quartet.corner_particles)
            ref_patches = top_patch_map[ref_corner.id]
            ref_corner.add_bond((angle_another, ref_patches['squareB'], closest_corner))
            ref_corner.add_bond((angle_another, ref_patches['squareA'], closest_corner))

        for ref_corner in center_quartet.corner_particles:
            closest_corner = self._nearest_corner(ref_corner, bottom_quartet.corner_particles)
            ref_patches = center_patch_map[ref_corner.id]
            ref_corner.add_bond((angle_another, ref_patches['squareB'], closest_corner))
            ref_corner.add_bond((angle_another, ref_patches['squareA'], closest_corner))

        for ref_corner in bottom_quartet.corner_particles:
            closest_corner = self._nearest_corner(ref_corner, center_quartet.corner_particles)
            ref_patches = bottom_patch_map[ref_corner.id]
            ref_corner.add_bond((angle_another, ref_patches['squareB'], closest_corner))
            ref_corner.add_bond((angle_another, ref_patches['squareA'], closest_corner))


    

    

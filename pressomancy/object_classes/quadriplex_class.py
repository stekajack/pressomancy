import espressomd
import numpy as np
from itertools import combinations, product
import random
import os
from pressomancy.helper_functions import load_coord_file, PartDictSafe
from pressomancy.object_classes.object_class import Simulation_Object 


class Quadriplex(metaclass=Simulation_Object):

    '''
    Class that contains quadriplex relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    numInstances = 0
    sigma = 5
    last_index_used = 0
    part_types = PartDictSafe({'real': 1, 'virt': 2})
    fene_handle = None
    bending_handle = None
    size=0.
    n_parts=3
    def __init__(self, sigma, quartet_grp, espresso_handle, fene_k=0., fene_r0=0., bending_k=0., bending_angle=None, bonding_mode=None, size=None):
        '''
        Initialisation of a quadriplex object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        assert isinstance(espresso_handle, espressomd.System)

        Quadriplex.sigma = sigma
        Quadriplex.sys = espresso_handle
        Quadriplex.bending_k = None
        Quadriplex.bending_angle = None
        Quadriplex.bonding_mode = bonding_mode
        if size==None:
            Quadriplex.size = np.sqrt(3)*Quadriplex.sigma
        else:
            Quadriplex.size = size


        # print('Qudariplex bonding mode: ', bonding_mode)
        if fene_k:
            if Quadriplex.fene_handle is None:
                Quadriplex.fene_k = fene_k
                Quadriplex.fene_r0 = fene_r0
                Quadriplex.fene_handle = espressomd.interactions.FeneBond(
                    k=Quadriplex.fene_k, r_0=Quadriplex.fene_r0, d_r_max=Quadriplex.fene_r0*1.5)
                Quadriplex.sys.bonded_inter.add(Quadriplex.fene_handle)
                print('Quadriplex set fene bond, should only happen once!!!')
        if bending_k:
            if Quadriplex.bending_handle is None:
                Quadriplex.bending_k = bending_k
                Quadriplex.bending_angle = bending_angle
                Quadriplex.bending_handle = espressomd.interactions.AngleHarmonic(
                    bend=Quadriplex.bending_k, phi0=Quadriplex.bending_angle)
                Quadriplex.sys.bonded_inter.add(Quadriplex.bending_handle)
                print('Quadriplex set bending potential, should only happen once!!!')

        self.who_am_i = Quadriplex.numInstances
        Quadriplex.numInstances += 1
        self.associated_quartets = quartet_grp
        self.realz_indices = []
        self.orientor = np.empty(shape=3, dtype=float)
        Quadriplex.n_parts = len(quartet_grp)

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. When the StopIteration except is caught Filament.last_index_used += Filament.n_parts . Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        assert Quadriplex.n_parts == 3, "a quadriplex can only be created from 3 quartets!!! "
        p_central = self.associated_quartets[0].set_object(
            pos, ori, triplet=self.who_am_i)
        self.realz_indices.extend(p_central)
        p_top = self.associated_quartets[1].set_object(pos+Quadriplex.fene_r0*ori,
                                                       ori, triplet=self.who_am_i)
        self.realz_indices.extend(p_top)
        p_bottom = self.associated_quartets[2].set_object(pos-Quadriplex.fene_r0*ori,
                                                          ori, triplet=self.who_am_i)
        self.realz_indices.extend(p_bottom)
        if Quadriplex.bonding_mode == 'ctc':
            self.bond_quartets_center_to_center()
            if Quadriplex.bending_k:
                self.add_bending_potential()
        if Quadriplex.bonding_mode == 'ftf':
            self.bond_quartets_corner_to_corner()
            if Quadriplex.bending_k:
                self.add_bending_potential()

        return self.realz_indices

    def bond_quartets_center_to_center(self):
        assert len(
            self.associated_quartets) == 3, "a quadriplex can only be created from 3 quartets!!! "
        self.sys.part.by_id(self.associated_quartets[0].realz_indices[0]).add_bond(
            (Quadriplex.fene_handle, self.sys.part.by_id(self.associated_quartets[1].realz_indices[0])))
        self.sys.part.by_id(self.associated_quartets[0].realz_indices[0]).add_bond(
            (Quadriplex.fene_handle, self.sys.part.by_id(self.associated_quartets[2].realz_indices[0])))

    def bond_quartets_corner_to_corner(self):
        assert len(
            self.associated_quartets) == 3, "a quadriplex can only be created from 3 quartets!!! "
        candidate1, candidate2, candidate3, pair_distances = self.associated_quartets[
            0].corner_particles, self.associated_quartets[1].corner_particles, self.associated_quartets[2].corner_particles, []

        pos1 = np.array([x.pos for x in candidate1])
        pos2 = np.array([x.pos for x in candidate2])
        pos3 = np.array([x.pos for x in candidate3])

        indices_int = np.arange(len(candidate1))
        index_combinations = np.array(
            list(product(indices_int, indices_int)))
        pair_distances = np.linalg.norm(
            pos1[index_combinations[:, 0]] - pos2[index_combinations[:, 1]], axis=-1)
        filtered_indices = index_combinations[np.isclose(
            pair_distances, self.fene_r0)]
        for index_pair in filtered_indices:
            p1_id, p2_id = index_pair
            self.sys.part.by_id(candidate1[p1_id].id).add_bond(
                (Quadriplex.fene_handle, self.sys.part.by_id(candidate2[p2_id].id)))

        pair_distances = np.linalg.norm(
            pos1[index_combinations[:, 0]] - pos3[index_combinations[:, 1]], axis=-1)
        filtered_indices = index_combinations[np.isclose(
            pair_distances, self.fene_r0)]
        for index_pair in filtered_indices:
            p1_id, p2_id = index_pair
            self.sys.part.by_id(candidate1[p1_id].id).add_bond(
                (Quadriplex.fene_handle, self.sys.part.by_id(candidate3[p2_id].id)))

    def add_bending_potential(self):
        assert len(
            self.associated_quartets) == 3, "a quadriplex can only be created from 3 quartets!!! "
        if Quadriplex.bonding_mode == 'ctc':
            central = self.sys.part.by_id(
                self.associated_quartets[0].realz_indices[0])
            top = self.sys.part.by_id(
                self.associated_quartets[1].realz_indices[0])
            bottom = self.sys.part.by_id(
                self.associated_quartets[2].realz_indices[0])
            central.add_bond(
                (Quadriplex.bending_handle, top,  bottom))
            if Quadriplex.bonding_mode == 'ftf':
                center_corners = self.associated_quartets[0].corner_particles
                top_corners = self.associated_quartets[1].corner_particles
                bottom_corners = self.associated_quartets[-1].corner_particles
                for central, top, bottom in zip(center_corners, top_corners, bottom_corners):
                    central.add_bond(
                        (Quadriplex.bending_handle, top,  bottom))

    def mark_covalent_bonds(self, part_type=666):
        assert len(
            self.associated_quartets) == 3, "a quadriplex can only be created from 3 quartets!!! "
        self.associated_quartets[1].mark_covalent_corner(
            part_type=part_type)
        self.associated_quartets[2].mark_covalent_corner(
            part_type=part_type)


class Quartet(metaclass=Simulation_Object):

    '''
    Class that contains quartet relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quartet. Therefore many relevant parameters are class specific, not instance specific.
    '''

    numInstances = 0
    sigma = 1
    part_types = PartDictSafe({'real': 1, 'virt': 2})
    last_index_used = 0
    current_dir = os.path.dirname(__file__)
    resources_dir = os.path.join(current_dir, '..', 'resources')
    resource_file = os.path.join(resources_dir, 'g_quartet_mesh_coordinates.txt')
    n_parts=25
    referece_sheet = load_coord_file(resource_file)
    type = 'solid'
    fene_handle = None
    size=0.

    def __init__(self, sigma, n_parts, type, espresso_handle, fene_k=0., fene_r0=0., size=None):
        '''
        Initialisation of a quartet object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        assert isinstance(espresso_handle, espressomd.System)
        assert type == 'solid' or type == 'broken'
        self.type = type
        if self.type == 'broken':
            Quartet.part_types.update({'circ': 8,
                  'squareA': 4, 'squareB': 5, 'cation': 7})
   
        Quartet.sigma = sigma
        Quartet.n_parts = n_parts
        if size==None:
            Quartet.size = np.sqrt(2)*np.sqrt( Quartet.n_parts)*Quartet.sigma
        else:
            Quartet.size = size
        Quartet.sys = espresso_handle
        self.who_am_i = Quartet.numInstances
        Quartet.numInstances += 1
        self.realz_indices = []
        self.virts_indices = []
        self.orientor = np.empty(shape=3, dtype=float)
        self.triplets_associated = None
        self.corner_particles = []
        if fene_k:
            if Quartet.fene_handle is None:
                Quartet.fene_k = fene_k
                Quartet.fene_r0 = fene_r0
                Quartet.fene_handle = espressomd.interactions.FeneBond(
                    k=Quartet.fene_k, r_0=Quartet.fene_r0, d_r_max=Quartet.fene_r0*1.5)
                Quartet.sys.bonded_inter.add(Quartet.fene_handle)
                print('Quartet set fene bond, should only happen once!!!')

    def set_object(self,  pos, ori, triplet=None):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part. Using a generator object with the particle enumeration logic, and a try catch paradigm. When the StopIteration except is caught Filament.last_index_used += Filament.n_parts . Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        positions = Quartet.referece_sheet+pos
        particles = list(Quartet.sys.part.add(type=[Quartet.part_types['virt']]*Quartet.n_parts,
                                              pos=positions))
        diag = np.sqrt(2)*4
        self.corner_particles = [part for part0, part in product(
            particles, particles) if np.isclose(np.linalg.norm(part0.pos - part.pos), diag, atol=1e-06)]
        if self.type == 'solid':
            particles[0].type = Quartet.part_types['real']
            w, d, h = 4, 4, 0
            rot_inertia_x = 1/12*(h**2+d**2)
            rot_inertia_y = 1/12*(w**2+h**2)
            rot_inertia_z = 1/12*(w**2+d**2)

            particles[0].rotation = (True, True, True)
            particles[0].rinertia = (
                rot_inertia_x, rot_inertia_y, rot_inertia_z)

            np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
                particles[0], particles[1:])
            particles[0].director = ori
            self.realz_indices.append(particles[0].id)
            self.virts_indices.extend([x.id for x in particles[1:]])

        if self.type == 'broken':
            particles[0].type = Quartet.part_types['real']
            particles[0].q = 5

            recepie_dict = {'assoc': {1: [2, 3, 6, 7, 8], 5: [4, 9, 10, 13, 14], 20: [11, 12, 15, 16, 21], 24: [
                17, 18, 19, 22, 23]}, 'circ': [8, 13, 12, 17], 'squareA': [6, 4, 21, 19], 'squareB': [11, 3, 22, 14], 'charged': [7, 9, 16, 18]}
            particles = np.array(particles)
            for part in self.corner_particles:
                part.type = Quartet.part_types['real']
                part.rotation = (True, True, True)
                part.director = ori
                self.realz_indices.append(part.id)

            for part in particles[np.array(recepie_dict['circ'])]:
                part.type = Quartet.part_types['circ']
                part.q = -1.25

            if Quartet.fene_k != None:
                for part1, part2 in zip(particles[np.array(recepie_dict['squareA'])], particles[np.array(recepie_dict['squareB'])]):
                    part1.type = Quartet.part_types['squareA']
                    part2.type = Quartet.part_types['squareB']
                    self.sys.part.by_id(part1.id).add_bond(
                        (Quartet.fene_handle, self.sys.part.by_id(part2.id)))

            else:

                for part in particles[np.array(recepie_dict['squareA'])]:
                    part.type = Quartet.part_types['squareA']

                for part in particles[np.array(recepie_dict['squareB'])]:
                    part.type = Quartet.part_types['squareB']

            for key, values in recepie_dict['assoc'].items():
                np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
                    particles[key], particles[values])
                for ii, jj in combinations([particles[key].id,]+[x.id for x in particles[values]], 2):
                    self.sys.part.by_id(ii).add_exclusion(jj)
                self.virts_indices.extend(
                    [part.id for part in particles[values]])
        if triplet is not None:
            self.triplets_associated = triplet
        Quadriplex.last_index_used += Quartet.n_parts
        return self.realz_indices

    def exclude_self_interactions(self):
        for ii, jj in combinations(self.realz_indices+self.virts_indices, 2):
            Quartet.sys.part.by_id(ii).add_exclusion(jj)

    def mark_covalent_corner(self, part_type=666):
        random_part = random.choice(self.corner_particles)
        random_part.type = part_type

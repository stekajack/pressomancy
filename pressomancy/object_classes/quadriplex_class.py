import espressomd
import numpy as np
from itertools import combinations, product
import random
import os
from pressomancy.helper_functions import load_coord_file, PartDictSafe, SinglePairDict, align_vectors
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
import warnings
import logging
from pressomancy.helper_functions import BondWrapper

class Quartet(metaclass=Simulation_Object):

    '''
    Class that contains quartet relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quartet. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    _resources_dir = os.path.join( os.path.dirname(__file__), '..', 'resources')
    _resource_file = os.path.join(_resources_dir, 'g_quartet_mesh_coordinates.txt')
    _referece_sheet = load_coord_file(_resource_file)
    simulation_type=SinglePairDict('quartet', 11)
    part_types = PartDictSafe({'real': 1, 'virt': 2})
    config = ObjectConfigParams(
        n_parts=len(_referece_sheet),
        type='solid', 
        bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0))
    )

    def __init__(self,config: ObjectConfigParams):
        '''
        Initialisation of a quartet object requires the specification of particle size, number of parts and a handle to the espresso system
        '''

        assert config['type'] == 'solid' or config['type'] == 'broken', 'type must be either solid or broken!!!'
        assert config['n_parts'] == len(Quartet._referece_sheet), 'n_parts must be equal to the number of parts in the reference sheet!!!'
        Quartet.numInstances += 1
        self.sys=config['espresso_handle']
        self.params=config
        if self.params['type'] == 'broken':
            assert self.params['bond_handle'] != None, 'broken quartets require a bond to be set!!!'
            Quartet.part_types.update({'circ': 28,
                  'squareA': 24, 'squareB': 25, 'cation': 27})
       
        self.who_am_i = Quartet.numInstances
        self.orientor = np.empty(shape=3, dtype=float)
        self.corner_particles = []
        self.type_part_dict=PartDictSafe({key: [] for key in Quartet.part_types.keys()})
        self.associated_objects=self.params['associated_objects']

    def set_object(self,  pos, ori, triplet=None):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part. Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        rotation_matrix = align_vectors(np.array([0,0,1]),ori) # 0,0,1 is the default director in espressomd
        positions = np.dot(Quartet._referece_sheet,rotation_matrix.T) + pos 
        particles=[self.add_particle(type_name='virt',pos=pos) for pos in positions]
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

        if self.params['type'] == 'broken':
            self.change_part_type(particles[0],'real')
            particles[0].q = 5

            recepie_dict = {'assoc': {1: [2, 3, 6, 7, 8], 5: [4, 9, 10, 13, 14], 20: [11, 12, 15, 16, 21], 24: [
                17, 18, 19, 22, 23]}, 'circ': [8, 13, 12, 17], 'squareA': [6, 4, 21, 19], 'squareB': [11, 3, 22, 14], 'charged': [7, 9, 16, 18]}
            particles = np.array(particles)
            for part in self.corner_particles:
                self.change_part_type(part,'real')
                part.rotation = (True, True, True)
                part.director = ori

            for part in particles[np.array(recepie_dict['circ'])]:
                self.change_part_type(part,'circ')
                part.q = -1.25

            if self.params['bond_handle'] != None:
                for part1, part2 in zip(particles[np.array(recepie_dict['squareA'])], particles[np.array(recepie_dict['squareB'])]):
                    self.change_part_type(part1,'squareA')
                    self.change_part_type(part2,'squareB')
                    self.bond_owned_part_pair(part1, part2)
            else:
                for part in particles[np.array(recepie_dict['squareA'])]:
                    self.change_part_type(part,'squareA')

                for part in particles[np.array(recepie_dict['squareB'])]:
                    self.change_part_type(part,'squareB')

            for key, values in recepie_dict['assoc'].items():
                np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
                    particles[key], particles[values])
                for ii, jj in combinations([particles[key].id,]+[x.id for x in particles[values]], 2):
                    self.sys.part.by_id(ii).add_exclusion(jj)
        return self

    def exclude_self_interactions(self):
        flattened_parts=np.array(self.type_part_dict.values()).flatten()
        for ii, jj in combinations(flattened_parts, 2):
           ii.add_exclusion(jj)
        warnings.warn(f'excluded self_interactions within {self.__class__.__name__}s')

    def mark_covalent_corner(self, part_type=666):
        random_part = random.choice(self.corner_particles)
        random_part.type = part_type
        logging.info(f'covalent corners marked for {self.__class__.__name__}s with part_type {part_type}')

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


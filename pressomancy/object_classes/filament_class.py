import espressomd
import numpy as np
import random
from itertools import product, pairwise
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import RoutineWithArgs, make_centered_rand_orient_point_array, PartDictSafe, SinglePairDict, BondWrapper
import logging
import warnings

class Filament(metaclass=Simulation_Object):
    '''
    Class that contains filament relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Filament. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    simulation_type=SinglePairDict('filament', 54)
    part_types = PartDictSafe({'real': 1, 'virt': 2})
    config = ObjectConfigParams(
        bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        spacing=None,)

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a filament object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        self.params=config
        self.associated_objects=self.params['associated_objects']
        self.build_function=RoutineWithArgs(
            func=make_centered_rand_orient_point_array,
            num_monomers=self.params['n_parts'],
            spacing=self.params['spacing'],
            )
        if self.associated_objects==None:
            monomer_size=(self.params['size']- self.params['bond_handle'].r_0*(self.params['n_parts']-1))*pow(self.params['n_parts'],-1)
            self.build_function.monomer_size=monomer_size
            warnings.warn('monomer size infered from Filament size and the BondWrapper.r_0')
        else:
            self.build_function.monomer_size=self.associated_objects[0].params['size']
        self.who_am_i = Filament.numInstances
        Filament.numInstances += 1
        self.orientor = np.empty(shape=3, dtype=float)
        self.type_part_dict=PartDictSafe({key: [] for key in Filament.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        pos=np.atleast_2d(pos)
        assert len(
            pos) == self.params['n_parts'], 'there is a missmatch between the pos lenth and Filament n_parts'
        self.orientor = ori
        if self.associated_objects is None:
            logic = (self.add_particle(type_name='real',pos=pp, rotation=(True, True, True)) for pp in pos)
        else:
            assert self.params['n_parts'] == len(
                self.associated_objects), " there doest seem to be enough monomers stored!!! "
            if not all(hasattr(obj, 'set_object') and callable(getattr(obj, 'set_object')) for obj in self.associated_objects):
                raise TypeError("One or more objects do not implement a callable 'set_object'")
            type_str=self.associated_objects[0].simulation_type.key
            logic = (obj_el.set_object(pos_el, self.orientor)
                        for obj_el, pos_el in zip(self.associated_objects, pos))
        for part in logic:
            pass
        return self

    def add_anchors(self,type_name):
        '''
        Adds virtual particles at top and bottom of a particle with size sigma, as determined by the orientation vector calculated by the get_orientation_vec(). Logic firstly adds front anchors and then back anchors, so there is a consistent logic to track ids. Indices of particles added here are stored in self.fronts_indices/self.backs_indices attributes respectively.

        :None:

        '''
        handles=[]
        if self.associated_objects!=None:
            assert all([type_name in x.part_types.keys() for x in self.associated_objects]), 'type key must exist in the part_types of all associated monomers!'
            warnings.warn('add_anchors should be used with caution for generic objects')
            for obj in self.associated_objects:
                handles.extend(obj.type_part_dict[type_name])
        else:
            handles = self.type_part_dict[type_name]
            
                
        self.fronts_indices=[]
        self.backs_indices=[]
        director = self.orientor
        for pp in handles:
            pp.director = director
        logic_front = ((self.add_particle(type_name='virt', pos=pp.pos + 0.5 * self.params['sigma'] * director, rotation=(False, False, False)), pp) for pp in handles)
        logic_back = ((self.add_particle(type_name='virt', pos=pp.pos - 0.5 * self.params['sigma'] * director, rotation=(False, False, False)), pp) for pp in handles)

        for p_hndl_front, pp in logic_front:
            p_hndl_front.vs_auto_relate_to(pp)
            self.fronts_indices.append(p_hndl_front.id)

        for p_hndl_back, pp in logic_back:
            p_hndl_back.vs_auto_relate_to(pp)
            self.backs_indices.append(p_hndl_back.id)
            # logging.info(f'anchors added for Filament {self.who_am_i}')

    def bond_overlapping_virtualz(self, crit=0.):
        '''
        Adds FENE bonds between virtuals that fulfill the crit distance criterion. In general, it is assumed that there are virtual anchors placed using the add_anchors() method, and that between two real parts one can always found a pair of either overlapping virts or at a distance corresponding to the FENE_r0 parameter (crit param can be arbitrary but should be realated to the aforementioned params). Relies on np.isclose().

        :return: None

        '''
        handles_font = list(self.sys.part.by_ids(self.fronts_indices))
        handles_back = list(self.sys.part.by_ids(self.backs_indices))
        for pp_f, pp_b in product(handles_font, handles_back):
            if np.isclose(np.linalg.norm(pp_f.pos-pp_b.pos), crit):
                self.bond_owned_part_pair(pp_f,pp_b)
        

    def add_dipole_to_embedded_virt(self, type_name, dip_magnitude=1.):
        '''
        Adds virtual particles to the center of each particle whose index is stored in self.realz_indices. It is critical that said virtuals do not have a director and have disabled rotation!

        :param dip_magnitude: float | magnitude of the dipole moment to be asigned using the self.orientor unit vector. Default=1.
        :return: None

        '''
        self.magnetizable_virts=[]
        Filament.dip_magnitude = dip_magnitude
        handles=[]
        self.__class__.part_types.update({'to_be_magnetized': 3})
        if self.associated_objects!=None:
            assert all([type_name in x.part_types.keys() for x in self.associated_objects]), 'type key must exist in the part_types of all associated monomers!'
            warnings.warn('add_anchors should be used with caution for generic objects')
            for obj in self.associated_objects:
                handles.extend(obj.type_part_dict[type_name])
        else:
            handles = self.type_part_dict[type_name]
        for pp in handles:
            p_hndl=self.add_particle(type_name='to_be_magnetized', pos=pp.pos,dip=Filament.dip_magnitude*self.orientor, rotation=(False, False, False))
            p_hndl.vs_auto_relate_to(pp)
            self.magnetizable_virts.append(p_hndl.id)

    def add_dipole_to_type(self, type_name, dip_magnitude=1.):
        '''
        Adds dipoles to real particles.

        :param dip_magnitude: float | magnitude of the dipole moment to be asigned using the self.orientor unit vector. Default=1.
        :return: None

        '''
        Filament.dip_magnitude = dip_magnitude
        handles = self.type_part_dict[type_name]
        for x in handles:
            x.dip = Filament.dip_magnitude*self.orientor

    def center_filament(self):
        list_parts =self.type_part_dict['real']
        ref_index = int(self.params['n_parts']*0.5)
        ref_pos = list_parts[ref_index].pos
        shift = ref_pos
        for elem in list_parts:
            elem.pos = elem.pos-shift
        self.sys.integrator.run(steps=0)
        logging.info('center_filament() moved parts')

    def bond_center_to_center(self, type_name):
        
        if self.associated_objects!=None:
            assert all([type_name in x.part_types.keys() for x in self.associated_objects]), 'type key must exist in the part_types of all associated monomers!'

            for el1,el2 in pairwise(self.associated_objects):
                for x,y in zip(el1.type_part_dict[type_name],el2.type_part_dict[type_name]):
                    self.bond_owned_part_pair(x,y)
        else:
            for x,y in pairwise(self.type_part_dict[type_name]):
                self.bond_owned_part_pair(x,y)

    def bond_nearest_part(self, type_name):
        '''
        Docstring for bond_nearest_part
        
        :param self: Description
        :type self:  
        :param bond_handle: Description
        :type bond_handle:  
        :param type_name: Description
        :type type_name:  '''
        assert all([type_name in x.part_types.keys() for x in self.associated_objects]), 'type key must exist in the part_types of all associated monomers!'
        assert self.associated_objects != None, 'self.associated_objects must not be None for this method ot work correctly'
        len_sq=pow(self.associated_objects[0].params['n_parts'],2)
        for el1,el2 in pairwise(self.associated_objects):
            el1_pos=np.mean([x.pos for x in el1.type_part_dict['real']],axis=0)
            el2_pos=np.mean([x.pos for x in el2.type_part_dict['real']],axis=0)
            midpoint = (el1_pos+el2_pos)*0.5
            small_spheres1 = sorted(el1.type_part_dict[type_name], key=lambda s: np.linalg.norm(s.pos - midpoint))
            small_spheres2 = sorted(el2.type_part_dict[type_name], key=lambda s: np.linalg.norm(s.pos - midpoint))

            x, y = small_spheres1[0], small_spheres2[0]
            self.bond_owned_part_pair(x,y)
    
    def add_bending_potential(self, type_name, bond_handle):
        flat_part_list=[]
        if self.associated_objects!=None:
            for obj in self.associated_objects:
                flat_part_list.extend(obj.type_part_dict[type_name])
        else:
            flat_part_list.extend(self.type_part_dict[type_name])

        if bond_handle not in self.sys.bonded_inter:
            self.sys.bonded_inter.add(bond_handle)
            logging.info(f'bond handle added to system for Object {self.__class__,self.who_am_i}')

        for iid in range(1,len(flat_part_list)-1):
            flat_part_list[iid].add_bond((bond_handle, flat_part_list[iid+1],  flat_part_list[iid-1]))
        
    def bond_quadriplexes(self, mode='hinge'):
        '''
        associated_objects contains monomer objects (assume quadriplex). We add cormer particles in each quadriplex pair to a pool of candidate corners: candidate1 and candidate2. Finaly checks which corner pairs have a distance self.params['sigma']-2*fene_r0. Relies on np.isclose().
        :return: None

        '''
        monomer_pairs = zip(self.associated_objects,
                            self.associated_objects[1:])
        for pair in monomer_pairs:
            fene_r0 = pair[0].params['bond_handle'].r_0
            candidate1, candidate2, pair_distances = [], [], []
            candidate1.extend(pair[0].associated_objects[1].corner_particles)
            candidate1.extend(pair[0].associated_objects[2].corner_particles)
            candidate2.extend(pair[-1].associated_objects[1].corner_particles)
            candidate2.extend(pair[-1].associated_objects[2].corner_particles)
            pos1 = np.array([x.pos for x in candidate1])
            pos2 = np.array([x.pos for x in candidate2])

            indices_int = np.arange(len(candidate1))
            index_combinations = np.array(
                list(product(indices_int, indices_int)))
            pair_distances = np.linalg.norm(
                pos1[index_combinations[:, 0]] - pos2[index_combinations[:, 1]], axis=-1)
            filtered_indices = index_combinations[np.isclose(
                pair_distances, fene_r0)]

            if mode == 'hinge':
                random_pair = random.choice(filtered_indices)
                self.bond_owned_part_pair(candidate1[random_pair[0]],candidate2[random_pair[-1]])

            if mode == 'all':
                for pair in filtered_indices:
                    self.bond_owned_part_pair(candidate1[pair[0]],candidate2[pair[-1]])

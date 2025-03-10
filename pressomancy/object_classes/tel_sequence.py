import espressomd
import numpy as np
import random
from pressomancy.object_classes.quadriplex_class import *
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import RoutineWithArgs, make_centered_rand_orient_point_array, PartDictSafe, SinglePairDict, BondWrapper
import logging
import warnings

def rule_maker(choice_id, offset, n=3):
    top = [26, 45, 49, 30]
    bottom = [51, 70, 74, 55]
    length = 4
    trigger_warning = False

    try:
        i = bottom.index(choice_id-offset)
    except ValueError:
        i = top.index(choice_id-offset)
        trigger_warning = True

    results = []
    free_end = 0
    for _ in range(n):
        next_index = (i + 1) % length
        if trigger_warning:
            results.append((bottom[i]+offset, top[next_index]+offset))
            free_end = bottom[next_index]+offset
        else:
            results.append((top[i]+offset, bottom[next_index]+offset))
            free_end = top[next_index]+offset

        i = (i + 1) % length
    return results, free_end

class TelSeq(metaclass=Simulation_Object):
    '''
    Class that contains TelSeq relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a TelSeq. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    simulation_type=SinglePairDict('tel_seq', 37)
    part_types = PartDictSafe({'real': 1, 'virt': 2,'to_be_magnetized':3})
    config = ObjectConfigParams(
        bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        diag_bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        spacing=None,
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a TelSeq object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        TelSeq.numInstances += 1
        self.sys=config['espresso_handle']
        self.params=config
        if self.params['associated_objects']==None:
            warnings.warn('no associated_objects have been passed explicity. Creating objects required to initialise object implicitly!')
            configuration=Quartet.config.specify(espresso_handle=self.sys,type='broken')
            quartets=[Quartet(config=configuration) for _ in range(3*self.params['n_parts'])]
            grouped_quartets = [quartets[i:i+3]
                    for i in range(0, len(quartets), 3)]
            quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=self.sys) for elem in grouped_quartets]
            self.params['associated_objects']= [Quadriplex(config=elem) for elem in quadriplex_config_list]
        self.associated_objects=self.params['associated_objects']

        self.build_function=RoutineWithArgs(func=make_centered_rand_orient_point_array,num_monomers=self.params['n_parts'],spacing=config['spacing'])  
        self.who_am_i = TelSeq.numInstances
        self.orientor = np.empty(shape=3, dtype=float)
        self.type_part_dict=PartDictSafe({key: [] for key in TelSeq.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of TelSeq stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        pos=np.atleast_2d(pos)
        assert len(
            pos) == self.params['n_parts'], 'there is a missmatch between the pos lenth and TelSeq n_parts'
        self.orientor = ori

        assert self.params['n_parts'] == len(
            self.associated_objects), " there doest seem to be enough monomers stored!!! "
        assert all([x.simulation_type==self.associated_objects[0].simulation_type for x in self.associated_objects[1:]]), 'all objects must have the same simulation type!'
        for obj_el, pos_el in zip(self.associated_objects, pos):
            _=obj_el.set_object(pos_el, self.orientor)
        return self

    def wrap_into_Tel(self):
        '''
        associated_objects contains monomer objects (assume quadriplex). We add cormer particles in each quadriplex pair to a pool of candidate corners: candidate1 and candidate2. Finaly checks which corner pairs have a distance self.params['sigma']-2*fene_r0. Relies on np.isclose().
        :return: None

        '''
        for iid in range(len(self.associated_objects)):
            monomer = self.associated_objects[iid]
            fene_r0 = self.params['bond_handle'].r_0
            candidates1 = []
            candidates1.extend(monomer.associated_objects[1].corner_particles)
            candidates1.extend(monomer.associated_objects[2].corner_particles)
            if monomer == self.associated_objects[0]:
                start_part_id = random.choice(candidates1).id
            logging.info(f'begin print {start_part_id}')
            res, free_end = rule_maker(start_part_id, monomer.who_am_i*75)
            logging.info(f'res, free_end {res, free_end}')
            for id1, id2 in res:
                self.bond_owned_part_pair(self.sys.part.by_id(id1), self.sys.part.by_id(id2), bond_handle=self.params['diag_bond_handle'])

            candidates2 = []

            try:
                monomer = self.associated_objects[iid+1]
                candidates2.extend(
                    monomer.associated_objects[1].corner_particles)
                candidates2.extend(
                    monomer.associated_objects[2].corner_particles)
                candidate_pos = np.array([x.pos for x in candidates2])

                pair_distances = np.linalg.norm(
                    candidate_pos-self.sys.part.by_id(free_end).pos, axis=-1)
                filtered = np.isclose(
                    pair_distances, fene_r0)

                if filtered.any() == True:
                    index = np.argmax(filtered)
                    self.bond_owned_part_pair(candidates2[index],self.sys.part.by_id(free_end))
                else:
                    logging.info('alt filter')
                    pair_distances = np.linalg.norm(
                        candidate_pos-self.sys.part.by_id(start_part_id).pos, axis=-1)
                    filtered = np.isclose(
                        pair_distances, fene_r0)
                    index = np.argmax(filtered)
                    self.bond_owned_part_pair(candidates2[index],self.sys.part.by_id(start_part_id))


                logging.info(f'index, start_part_id {res, free_end}')

                start_part_id = candidates2[index].id
                logging.info(f'end print {start_part_id}')
            except IndexError:
                logging.info('end of chain rached')
                continue

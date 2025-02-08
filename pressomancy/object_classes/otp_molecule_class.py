import espressomd
import os
from pressomancy.helper_functions import load_coord_file, PartDictSafe, SinglePairDict, align_vectors,BondWrapper
import numpy as np
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 


class OTP(metaclass=Simulation_Object):

    '''
    Class that contains OTP relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a OTP. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0

    _resources_dir = os.path.join( os.path.dirname(__file__), '..', 'resources')
    _resource_file = os.path.join(_resources_dir, 'otp_coordinates.txt')
    _referece_sheet = load_coord_file(_resource_file)[1:]
    _referece_sheet-=np.mean(_referece_sheet, axis=0)
    simulation_type= SinglePairDict('otp', 6)
    part_types = PartDictSafe(simulation_type)
    config = ObjectConfigParams(
        n_parts=len(_referece_sheet),
        sigma=0.483,
        long_side=0.7191944807239401,
        rig_bond_long = BondWrapper(espressomd.interactions.RigidBond(r=0.7191944807239401, ptol=1e-12, vtol=1e-12)), 
        rig_bond_short = BondWrapper(espressomd.interactions.RigidBond(r=0.483, ptol=1e-12, vtol=1e-12))
    ) 


    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a crowder object requires the specification of particle size and a handle to the espresso system
        '''
        assert config['n_parts'] == len(OTP._referece_sheet), 'n_parts must be equal to the number of parts in the reference sheet!!!'
        self.sys=config['espresso_handle']
        self.params=config
        OTP.numInstances += 1
        self.who_am_i = OTP.numInstances
        self.realz_indices = []
        self.virts_indices = []
        self.associated_objects=self.params['associated_objects']        
        self.type_part_dict=PartDictSafe({key: [] for key in OTP.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. When the StopIteration except is caught Filament.last_index_used += Filament.n_parts . Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        rotation_matrix = align_vectors(np.array([0.0,0.0,1.0]),ori) # 0,0,1 is the default director in espressomd
        rotated_rigid_body = np.dot(OTP._referece_sheet,rotation_matrix.T) + np.tile(pos, (self.params['n_parts'],1))
        parts = [self.add_particle(type_name='otp', pos=el_pos) for el_pos in rotated_rigid_body]
        self.bond_owned_part_pair(parts[0],parts[1], bond_handle=self.params['rig_bond_long'])
        self.bond_owned_part_pair(parts[0],parts[-1], bond_handle=self.params['rig_bond_short'])
        self.bond_owned_part_pair(parts[1],parts[-1], bond_handle=self.params['rig_bond_short'])
        parts[0].add_exclusion(parts[1])
        parts[0].add_exclusion(parts[2])
        parts[1].add_exclusion(parts[2])
        self.realz_indices.extend([particle.id for particle in parts])
        return [particle.id for particle in parts]

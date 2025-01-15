import espressomd
import os
from pressomancy.helper_functions import load_coord_file, PartDictSafe, SinglePairDict
import numpy as np
from pressomancy.object_classes.object_class import Simulation_Object 
from scipy.spatial.transform import Rotation as R


class OTP(metaclass=Simulation_Object):

    '''
    Class that contains OTP relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a OTP. Therefore many relevant parameters are class specific, not instance specific.
    '''

    numInstances = 0
    sigma: float = 1. # particle size used mostly by the Filament class rn

    last_index_used = 0
    n_parts = 3
    long_side=(None,0.719194480723940)
    short_side=(None,0.483)
    _resources_dir = os.path.join( os.path.dirname(__file__), '..', 'resources')
    _resource_file = os.path.join(_resources_dir, 'otp_coordinates.txt')
    _referece_sheet = load_coord_file(_resource_file)[1:]
    _referece_sheet-=np.mean(_referece_sheet, axis=0)
    size=0.
    simulation_type= SinglePairDict('otp', 6)
    part_types = PartDictSafe(simulation_type)

    def __init__(self, sigma, long_side, short_side, espresso_handle,associated_objects=None, size=None):
        '''
        Initialisation of a crowder object requires the specification of particle size and a handle to the espresso system
        '''
        assert isinstance(espresso_handle, espressomd.System)
        OTP.numInstances += 1
        OTP.sigma = sigma
        OTP.sys = espresso_handle
        self.who_am_i = OTP.numInstances
        OTP.long_side=long_side
        OTP.short_side=short_side
        self.realz_indices = []
        self.virts_indices = []
        if size==None:
            OTP.size=OTP.long_side[1]+OTP.sigma
        else:
            OTP.size=size
        self.associated_objects=associated_objects
        self.type_part_dict=PartDictSafe({key: [] for key in OTP.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. When the StopIteration except is caught Filament.last_index_used += Filament.n_parts . Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        
        random_rotation = R.random()
        rotated_rigid_body = random_rotation.apply(OTP._referece_sheet)+np.tile(pos, (OTP.n_parts,1)) # type: ignore
        parts = list(OTP.sys.part.add(
            type=[OTP.part_types['otp'],]*OTP.n_parts, pos=rotated_rigid_body))
        parts[0].add_bond((OTP.long_side[0], parts[1]))
        parts[0].add_bond((OTP.short_side[0], parts[-1]))
        parts[1].add_bond((OTP.short_side[0], parts[-1]))
        parts[0].add_exclusion(parts[1])
        parts[0].add_exclusion(parts[2])
        parts[1].add_exclusion(parts[2])
        OTP.last_index_used += OTP.n_parts
        self.realz_indices.extend([particle.id for particle in parts])
        return [particle.id for particle in parts]

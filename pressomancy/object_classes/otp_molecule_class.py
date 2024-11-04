import espressomd
import os
from pressomancy.helper_functions import load_coord_file, PartDictSafe
import numpy as np
from pressomancy.object_classes.object_class import Simulation_Object 
from scipy.spatial.transform import Rotation as R


class OTP(metaclass=Simulation_Object):

    '''
    Class that contains OTP relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a OTP. Therefore many relevant parameters are class specific, not instance specific.
    '''

    numInstances = 0
    sigma = 1
    part_types = PartDictSafe({'otp': 1,})
    last_index_used = 0
    n_parts = 3
    long_side=(None,0.719194480723940)
    short_side=(None,0.483)
    referece_sheet=None
    size=0.

    def __init__(self, sigma, long_side, short_side, espresso_handle, size=None):
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
        current_dir = os.path.dirname(__file__)
        resources_dir = os.path.join(current_dir, '..', 'resources')
        resource_file = os.path.join(resources_dir, 'otp_coordinates.txt')
        if OTP.referece_sheet is None:
            OTP.referece_sheet = load_coord_file(resource_file)[1:]
            OTP.referece_sheet=OTP.referece_sheet-np.mean(OTP.referece_sheet, axis=0)

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. When the StopIteration except is caught Filament.last_index_used += Filament.n_parts . Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        
        random_rotation = R.random()
        rotated_rigid_body = random_rotation.apply(OTP.referece_sheet)+np.tile(pos, (OTP.n_parts,1)) # type: ignore
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

from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import PartDictSafe, SinglePairDict

class PointDipolePermanent(metaclass=Simulation_Object):
    '''
    Class that contains permanent magnetic point dipole particles relevant paramaters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a PointDipolePermanent. Therefore many relevant parameters are class specific, not instance specific.
    '''

    required_features=['DIPOLES']	
    numInstances = 0
    simulation_type= SinglePairDict('point_dipole_permanent', 3)
    part_types = PartDictSafe({'pdp_real': 61})
    config = ObjectConfigParams(
         dipm=1.
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a PointDipolePermanent object requires the specification of particle size and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        self.params=config
        PointDipolePermanent.numInstances += 1
        self.who_am_i = PointDipolePermanent.numInstances
        self.associated_objects=config['associated_objects']
        self.type_part_dict=PartDictSafe({key: [] for key in PointDipolePermanent.part_types.keys()})
        assert self.associated_objects is None, "Point dipoles can not have associated objects. They are singular particles, as basic as possible."

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        dipm= self.params['dipm']
        hndl = self.add_particle(type_name='pdp_real', pos=pos, rotation=[True, True, True], dip=(dipm * ori))

        # Very Important Particle. To use to bond, calculate distances, and other Very Important Things. Usually at the center of mass, and usually a real particle
        self.vip = hndl

        return self
    
class PointDipoleSuperpara(metaclass=Simulation_Object):
    '''
    Class that contains superparamagnetic point dipole particles relevant paramaters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a PointDipoleSuperpara. Therefore many relevant parameters are class specific, not instance specific.

    Requires to run a magnetization function every step. Eg. system.magnetize()
    '''

    required_features=['DIPOLES', 'DIPOLE_FIELD_TRACKING']	
    numInstances = 0
    simulation_type= SinglePairDict('point_dipole_superpara', 4)
    part_types = PartDictSafe({'pds_real': 62, 'pds_virt': 666})
    config = ObjectConfigParams(
         dipm=1.
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a PointDipoleSuperpara object requires the specification of particle size and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        self.params=config
        PointDipoleSuperpara.numInstances += 1
        self.who_am_i = PointDipoleSuperpara.numInstances
        self.associated_objects=config['associated_objects']
        self.type_part_dict=PartDictSafe({key: [] for key in PointDipoleSuperpara.part_types.keys()})
        assert self.associated_objects is None, "Point dipoles can not have associated objects. They are singular particles, as basic as possible."

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        particl_real=self.add_particle(type_name='pds_real', pos=pos, rotation=[True, True, True], director=ori)
        
        particl_virt=self.add_particle(type_name='pds_virt', pos=pos, rotation=[False, False, False], dip=(ori * 1e-6))
        particl_virt.vs_auto_relate_to(particl_real)

        # Very Important Particle. To use to bond, calculate distances, and other Very Important Things. Usually at the center of mass, and usually a real particle
        self.vip = particl_real

        return self
    
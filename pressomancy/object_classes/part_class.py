from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import PartDictSafe, SinglePairDict

class GenericPart(metaclass=Simulation_Object):

    '''
    Class that contains quadriplex relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    simulation_type= SinglePairDict('generic_particle', 42)
    part_types = PartDictSafe({'real': 1,'virt': 2})
    config = ObjectConfigParams(
        espresso_part_kwargs=dict(),
        alias=None
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a crowder object requires the specification of particle size and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        self.params=config
        self.associated_objects=self.params['associated_objects']
        self.who_am_i = GenericPart.numInstances
        GenericPart.numInstances += 1
        self.type_part_dict=PartDictSafe({key: [] for key in GenericPart.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        particle=self.add_particle(type_name='real', pos=pos, rotation=(True, True, True), **self.params['espresso_part_kwargs'])
        particle.director = ori
        return self

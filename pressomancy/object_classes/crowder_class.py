import espressomd
from pressomancy.object_classes.object_class import Simulation_Object 
from pressomancy.helper_functions import PartDictSafe, SinglePairDict

class Crowder(metaclass=Simulation_Object):

    '''
    Class that contains quadriplex relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''

    numInstances = 0
    sigma = 1
    n_parts = 1
    size=0.
    simulation_type= SinglePairDict('crowder', 5)
    part_types = PartDictSafe({'crowder': 5})

    def __init__(self, sigma, espresso_handle, associated_objects=None,size=None):
        '''
        Initialisation of a crowder object requires the specification of particle size and a handle to the espresso system
        '''
        assert isinstance(espresso_handle, espressomd.System)
        Crowder.numInstances += 1
        Crowder.sigma = sigma
        if size==None:
            Crowder.size = Crowder.sigma
        else:
            Crowder.size = size
        Crowder.sys = espresso_handle
        self.who_am_i = Crowder.numInstances
        self.associated_objects=associated_objects
        self.type_part_dict=PartDictSafe({key: [] for key in Crowder.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        particle=self.add_particle(type_name='crowder', pos=pos, rotation=(False, False, False))
        particle.director = ori
        return self

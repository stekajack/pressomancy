import espressomd
from pressomancy.object_classes.object_class import Simulation_Object 
from pressomancy.helper_functions import PartDictSafe, SinglePairDict

class EGGPart(metaclass=Simulation_Object):

    '''
    Class that contains quadriplex relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=['MAGNETODYNAMICS_EGG_MODEL',]	
    numInstances = 0
    sigma = 1
    n_parts = 1
    size=0.
    simulation_type= SinglePairDict('egg_part', 74)
    part_types = PartDictSafe({'real': 1,'virt': 2})

    def __init__(self, sigma, espresso_handle, dipm, egg_gamma, aniso_energy, associated_objects=None,size=None):
        '''
        Initialisation of a EGGPart object requires the specification of particle size and a handle to the espresso system
        '''
        assert isinstance(espresso_handle, espressomd.System)
        EGGPart.numInstances += 1
        EGGPart.sigma = sigma
        if size==None:
            EGGPart.size = EGGPart.sigma
        else:
            EGGPart.size = size
        EGGPart.sys = espresso_handle
        self.egg_gamma=egg_gamma
        self.aniso_energy=aniso_energy
        self.dipm=dipm
        self.who_am_i = EGGPart.numInstances
        self.associated_objects=associated_objects
        self.type_part_dict=PartDictSafe({key: [] for key in EGGPart.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        particl_real=self.add_particle(type_name='real', pos=pos, rotation=(True, True, True), director=ori)

        particl_virt=self.add_particle(type_name='virt', pos=pos, rotation=(True, True, True), dipm=self.dipm, egg_model_params = (True, self.egg_gamma,self.aniso_energy))
        particl_virt.vs_auto_relate_to(particl_real)

        return self

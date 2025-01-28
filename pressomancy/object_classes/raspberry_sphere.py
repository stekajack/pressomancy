import espressomd
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.helper_functions import PartDictSafe, SinglePairDict, load_coord_file
import os
import numpy as np

class RaspberrySphere(metaclass=Simulation_Object):

    '''
    Class that contains relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    _resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
    _resource_file = os.path.join(_resources_dir, 'dungeon_witch_raspberry.txt')
    _referece_sheet = load_coord_file(_resource_file)
    simulation_type= SinglePairDict('raspberry', 69)
    part_types = PartDictSafe({'real': 1,'virt':2})
    config=ObjectConfigParams(
        n_parts=len(_referece_sheet),
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a RaspberrySphere object requires the specification of particle size and a handle to the espresso system
        '''
     
        assert config['n_parts'] == len(RaspberrySphere._referece_sheet), 'n_parts must be equal to the number of parts in the resource file!!!'
        self.sys=config['espresso_handle']
        self.params=config
        self.associated_objects=config['associated_objects']
        RaspberrySphere.numInstances += 1
        self.who_am_i = RaspberrySphere.numInstances
        self.type_part_dict=PartDictSafe({key: [] for key in RaspberrySphere.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        positions = RaspberrySphere._referece_sheet+pos
        particles=[self.add_particle(type_name='virt',pos=pos) for pos in positions]
        
        self.change_part_type(particles[0],'real')
        particles[0].rotation = (True, True, True)
        np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
            particles[0], particles[1:])
        particles[0].director = ori
       

        return self
    
    def set_hydrod_props(self,rot_inertia ,mass):
        for part in self.type_part_dict['real']:
            part.rinertia = np.ones(3) *rot_inertia
            part.mass = mass


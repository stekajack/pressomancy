from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.helper_functions import PartDictSafe, SinglePairDict, load_coord_file
import os
import numpy as np

class GenericRigidObj(metaclass=Simulation_Object):

    '''
    Class that contains relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=['VIRTUAL_SITES_RELATIVE',]
    numInstances = 0
    _resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
    _resource_file = {}
    _referece_sheet = {}

    simulation_type= SinglePairDict('generic_rigid_object', 68)
    part_types = PartDictSafe({'real': 1,'virt':2})
    config=ObjectConfigParams(
        n_parts=None,
        alias='raspberry_sphere'
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a GenericRigidObj object requires the specification of particle size and a handle to the espresso system
        '''
        assert config['alias'] is not None, 'Generic rigid object has to have an alias set, becaouse it is used to infer the correct resource file in resources!'
        _resource_file=GenericRigidObj._resource_file.get(config['alias'])
        if _resource_file is None:
            GenericRigidObj._resource_file[config['alias']]=os.path.join(self._resources_dir, f"{config['alias']}.txt")
            GenericRigidObj._referece_sheet[config['alias']]=load_coord_file(GenericRigidObj._resource_file[config['alias']])
        config['n_parts'] = len(GenericRigidObj._referece_sheet[config['alias']])
        self.params=config
        self.sys=config['espresso_handle']
        self.associated_objects=config['associated_objects']
        self.who_am_i = GenericRigidObj.numInstances
        GenericRigidObj.numInstances += 1
        self.type_part_dict=PartDictSafe({key: [] for key in GenericRigidObj.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        positions = GenericRigidObj._referece_sheet[self.params['alias']]+pos
        particles=[self.add_particle(type_name='virt',pos=pos) for pos in positions]
        
        self.change_part_type(particles[0],'real')
        particles[0].rotation = (True, True, True)
        np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
            particles[0], particles[1:])
        particles[0].director = ori
       

        return self

import espressomd
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import PartDictSafe, SinglePairDict

class SWPart(metaclass=Simulation_Object):

    '''
    Class that contains quadriplex relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    # required_features=['MAGNETODYNAMICS_TSW_MODEL',]	
    required_features=list()

    numInstances = 0
    simulation_type= SinglePairDict('sw_part', 13)
    part_types = PartDictSafe({'sw_real': 9,'sw_virt': 10})
    config = ObjectConfigParams(
        kT_KVm_inv=5.,
        dipm=1.75,
        dt_incr=1e-10,
        tau0_inv=7.35e+08,
        HK_inv=0.175,
        tau_trans_inv=0
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a SWPart object requires the specification of particle size and a handle to the espresso system
        '''
        
        self.sys=config['espresso_handle']
        self.params=config
        self.associated_objects=self.params['associated_objects']
        self.type_part_dict=PartDictSafe({key: [] for key in SWPart.part_types.keys()})
        self.who_am_i = SWPart.numInstances
        SWPart.numInstances += 1

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        particl_real=self.add_particle(type_name='sw_real', pos=pos, rotation=(True, True, True),kT_KVm_inv=self.params['kT_KVm_inv'],dt_incr=self.params['dt_incr'],tau0_inv=self.params['tau0_inv'], tau_trans_inv=self.params['tau_trans_inv'], director=ori,sw_real=True)

        particl_virt=self.add_particle(type_name='sw_virt', pos=pos, rotation=(False, False, False), sw_virt=True, Hkinv=self.params['HK_inv'],
            sat_mag=self.params['dipm'], dip=ori*self.params['dipm'])
        particl_virt.vs_auto_relate_to(particl_real)

        return self

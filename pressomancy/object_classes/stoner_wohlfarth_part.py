import espressomd
from pressomancy.object_classes.object_class import Simulation_Object 
from pressomancy.helper_functions import PartDictSafe, SinglePairDict

class SWPart(metaclass=Simulation_Object):

    '''
    Class that contains quadriplex relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''

    numInstances = 0
    sigma = 1
    n_parts = 1
    size=0.
    simulation_type= SinglePairDict('sw_part', 13)
    part_types = PartDictSafe({'sw_real': 9,'sw_virt': 10})

    def __init__(self, sigma, espresso_handle, kT_KVm_inv, dipm, dt_incr, tau0_inv, HK_inv, associated_objects=None,size=None,tau_trans_inv=0):
        '''
        Initialisation of a SWPart object requires the specification of particle size and a handle to the espresso system
        '''
        assert isinstance(espresso_handle, espressomd.System)
        SWPart.numInstances += 1
        SWPart.sigma = sigma
        if size==None:
            SWPart.size = SWPart.sigma
        else:
            SWPart.size = size
        SWPart.sys = espresso_handle
        self.kT_KVm_inv=kT_KVm_inv
        self.dt_incr=dt_incr
        self.tau0_inv=tau0_inv
        self.tau_trans_inv=tau_trans_inv
        self.HK_inv=HK_inv
        self.dipm=dipm
        self.who_am_i = SWPart.numInstances
        self.associated_objects=associated_objects
        self.type_part_dict=PartDictSafe({key: [] for key in SWPart.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        particl_real=self.add_particle(type_name='sw_real', pos=pos, rotation=(True, True, True),kT_KVm_inv=self.kT_KVm_inv,dt_incr=self.dt_incr,tau0_inv=self.tau0_inv, tau_trans_inv=self.tau_trans_inv, director=ori,sw_real=True)

        particl_virt=self.add_particle(type_name='sw_virt', pos=pos, rotation=(False, False, False), sw_virt=True, Hkinv=self.HK_inv,
            sat_mag=self.dipm, dip=ori*self.dipm)
        particl_virt.vs_auto_relate_to(particl_real)

        return self

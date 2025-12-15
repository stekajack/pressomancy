import numpy as np
from pressomancy.object_classes.object_class import ObjectConfigParams
from pressomancy.object_classes.rigid_obj import GenericRigidObj
from pressomancy.helper_functions import PartDictSafe

class RaspberrySphere(GenericRigidObj):

    '''
    Class that contains relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    numInstances = 0
    part_types = PartDictSafe()
    config=ObjectConfigParams(
        alias='raspberry_sphere'
    )
    def __init__(self, config: ObjectConfigParams):
        super().__init__(config)
        self.who_am_i = RaspberrySphere.numInstances
        RaspberrySphere.numInstances += 1
    
    def set_hydrod_props(self,rot_inertia ,mass):
        for part in self.type_part_dict['real']:
            part.rinertia = np.ones(3) *rot_inertia
            part.mass = mass


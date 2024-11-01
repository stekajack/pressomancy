import espressomd
from pressomancy.helper_functions import RoutineWithArgs

class Simulation_Object():

    '''
    Generic object class from which every object implementation should inherit. Containes common interface methods for any object to mimic virtual function behaviour.
    '''
    who_am_i=None
    
    def __eq__(self, other):
        # Ensure the same type and then compare based on `who_am_i`
        return isinstance(other, self.__class__) and self.who_am_i == other.who_am_i

    def __hash__(self):
        # Hash using class type and `who_am_i`
        return hash((self.__class__, self.who_am_i))
    
    build_function=RoutineWithArgs()

    def set_object(self,  pos, ori):
        raise NotImplementedError()

from pressomancy.helper_functions import RoutineWithArgs, PartDictSafe, SinglePairDict
import types

def generic_type_exception(name, attribute_name, expected_type):
    raise NotImplementedError(
                    f"The class attribute '{attribute_name}' is required in '{name}' but not defined. "
                    f"Please define '{attribute_name}' as a '{expected_type.__name__}' in your subclass."
                )

def generic_type_exception_inst(name, attribute_name, expected_type):
    raise NotImplementedError(
                    f"The instance attribute '{attribute_name}' is required in '{name}' but not defined. "
                    f"Please define '{attribute_name}' as a '{expected_type.__name__}' in your subclass."
                )
def generic_modify_system_attribute(self, clas_self, attribute_name,action):
    raise NotImplementedError('the reference to a manager class has not been set')

def helper_set_attribute(instance,attr_name,target):
    if not hasattr(instance,attr_name):
        bound_method = types.MethodType(target, instance)
        setattr(instance, attr_name, bound_method)

class Simulation_Object(type):
    """
    Metaclass for simulation objects. Enforces required attributes and provides shared methods.
    """
    def __init__(cls, name, bases, class_dict):
        super().__init__(name, bases, class_dict)
        # Assign class-level __iter__, __eq__, and __hash__ to make them work consistently
        cls.__iter__ = Simulation_Object._cusiter
        cls.__eq__ = Simulation_Object._eq
        cls.__hash__ = Simulation_Object._hash
        required_attributes = {
            "numInstances": int,
            "n_parts": int,
            "size": float,
            "part_types": PartDictSafe,
            "simulation_type": SinglePairDict
        }
        for attr, expected_type in required_attributes.items():
            if not hasattr(cls, attr):
                generic_type_exception(name, attr, expected_type)
            # Check if attribute is of the correct type
            elif not isinstance(getattr(cls, attr), expected_type):
                generic_type_exception(name, attr, expected_type)
    
    def __call__(cls, *args, **kwargs):
        # Create a new instance of the class
        instance = super().__call__(*args, **kwargs)
        # Assign the build_function to the class if it does not already have one

        helper_set_attribute(instance,"build_function",RoutineWithArgs())
        helper_set_attribute(instance,"set_object",Simulation_Object.set_object)
        helper_set_attribute(instance,"add_particle",Simulation_Object.add_particle)
        helper_set_attribute(instance,"change_part_type",Simulation_Object.change_part_type)
        helper_set_attribute(instance,"modify_system_attribute",generic_modify_system_attribute)

        required_attributes = {"who_am_i": int,'type_part_dict'  :PartDictSafe }
        for attr, expected_type in required_attributes.items():
            # Check for required instance attribute `who_am_i`
            if not hasattr(instance, attr):
                generic_type_exception_inst(instance.__class__.__name__, attr, expected_type)
        return instance

    # Define __eq__, __hash__, and __iter__ as class-level methods for consistency
    @staticmethod
    def _eq(self, other):
        if not isinstance(other, self.__class__):
            return False
        return getattr(self, "who_am_i", None) == getattr(other, "who_am_i", None)

    @staticmethod
    def _hash(self):
        return hash((self.__class__, getattr(self, "who_am_i", None)))

    @staticmethod
    def _cusiter(self):
        """Return an iterator for the internal list, making the object iterable."""
        return iter([self])

    # Shared method for setting object attributes (meant to be overridden if necessary)
    def set_object(self, *args,**kwargs):
        print(f"Self is: {self}")
        raise NotImplementedError("The 'set_object' method must be implemented in subclasses.")
    
    def add_particle(self, type_name, pos, **kwargs):
        '''
        Adds a particle to simualtion box, usign the espresso system.part.add() comand. Makes sure that every time a particle is added, 
        
        :param self: class instance
        :type self:class managed by Simulation_Object metaclass  
        :param type_name: name of the type from self.part_types
        :type type_name:  string
        :param pos: positon at which the particle is craeted
        :type pos: iterable shape=(N,3)
        :param kwargs: other properties as defined by the espresso api
        :return: returns a particle handle
        :rtype: ParticleHandle object'''

        assert type_name in self.part_types.keys(), 'an object can only add partices of types declared in self.part_types'
        part_params={'type':self.part_types[type_name],'pos': pos}
        part_params.update(**kwargs)

        part_hndl = self.sys.part.add(**part_params)
        self.type_part_dict[type_name].append(part_hndl)
        self.modify_system_attribute(self,'part_types', lambda current_value: current_value.update({type_name: part_params['type']}))
        return part_hndl
    
    def change_part_type(self, particle, new_type_name):
        assert new_type_name in self.part_types.keys(), 'an object can only add partices of types declared in self.part_types'
        current_key = next(key for key, values in self.type_part_dict.items() if particle in values)
        self.type_part_dict[current_key].remove(particle)
        particle.type=self.part_types[new_type_name]
        self.type_part_dict[new_type_name].append(particle)
        self.modify_system_attribute(self,'part_types', lambda current_value: current_value.update({new_type_name: self.part_types[new_type_name]}))


        
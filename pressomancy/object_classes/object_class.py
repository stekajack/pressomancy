from pressomancy.helper_functions import RoutineWithArgs, PartDictSafe

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

class Simulation_Object(type):
    """
    Metaclass for simulation objects. Enforces required attributes and provides shared methods.
    """
    def __init__(cls, name, bases, class_dict):
        super().__init__(name, bases, class_dict)
        # dict of required attributes
        required_attributes = {
            "numInstances": int,
            "n_parts": int,
            "size": float,
            "part_types": PartDictSafe
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
        if not hasattr(instance, "build_function"):
            instance.build_function =RoutineWithArgs()
        required_attributes = {"who_am_i": int,}
        for attr, expected_type in required_attributes.items():
            # Check for required instance attribute `who_am_i`
            if not hasattr(instance, attr):
                generic_type_exception_inst(instance.__class__.__name__, attr, expected_type)
        return instance

    # Shared method for equality comparison based on `who_am_i`
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return getattr(self, "who_am_i", None) == getattr(other, "who_am_i", None)

    # Shared method for hashing based on `who_am_i`
    def __hash__(self):
        return hash((self.__class__, getattr(self, "who_am_i", None)))

    # Placeholder for build function
    # Shared method for setting object attributes (meant to be overridden if necessary)
    def set_object(self, pos, ori):
        raise NotImplementedError("The 'set_object' method must be implemented in subclasses.")
from pressomancy.helper_functions import RoutineWithArgs, PartDictSafe, SinglePairDict
import types
from functools import partial
import logging
import espressomd
import itertools

def _generic_type_exception(scope, name, attribute_name, expected_type):
    raise NotImplementedError(
        f"The {scope} attribute '{attribute_name}' is required in '{name}' but not defined. "
        f"Please define '{attribute_name}' as a '{expected_type.__name__}' in your subclass."
    )

# Create partials for class and instance exceptions
generic_type_exception = partial(_generic_type_exception, "class")
generic_type_exception_inst = partial(_generic_type_exception, "instance")

def generic_modify_system_attribute(self, clas_self, attribute_name,action):
    raise NotImplementedError('the reference to a manager class has not been set')

def helper_set_attribute(instance,attr_name,target):
    if not hasattr(instance,attr_name):
        bound_method = types.MethodType(target, instance)
        setattr(instance, attr_name, bound_method)

class Simulation_Object(type):
    """
    Metaclass for simulation objects, providing attribute enforcement and shared methods.

    The `Simulation_Object` metaclass ensures that all simulation object subclasses adhere to a
    defined structure by enforcing required attributes at both the class and instance levels.
    It also provides common methods for particle management and system interaction.

    Key Features
    ------------
    - **Attribute enforcement**: Ensures subclasses define required attributes with the correct types.
    - **Default behavior**: Assigns shared methods and default behaviors to instances.
    - **Polymorphism**: Facilitates seamless integration of different object classes into the simulation framework.
    - **Object identification**: Implements equality and hashing based on unique instance attributes.

    Class-Level Required Attributes
    --------------------------------
    - `required_features` : list
        List of required features for the simulation object.
    - `numInstances` : int
        Tracks the number of instances of the class.
    - `part_types` : PartDictSafe
        Dictionary-like object mapping particle types to their identifiers.
    - `simulation_type` : SinglePairDict
        A single key-value pair representing the type of simulation object.

    Instance-Level Required Attributes
    -----------------------------------
    - `who_am_i` : int
        Unique identifier for the instance.
    - `type_part_dict` : PartDictSafe
        Tracks particle handles grouped by type.
    - `associated_objects` : list
        List of related simulation objects, if any.

    Methods
    -------
    __init__(cls, name, bases, class_dict):
        Enforces required class-level attributes during class creation.

    __call__(cls, *args, **kwargs):
        Creates a new instance and assigns default methods and attributes.

    set_object(self, *args, **kwargs):
        Abstract method for setting up simulation objects. Must be implemented by subclasses.

    delete_owned_parts(self):
        Deletes all particles owned by the object, including those in associated objects.

    add_particle(self, type_name, pos, **kwargs):
        Adds a particle to the simulation box, ensuring it adheres to the object's declared particle types.

    change_part_type(self, particle, new_type_name):
        Changes the type of an existing particle, updating all relevant tracking structures.

    _eq(self, other):
        Compares two instances for equality based on their `who_am_i` attribute.

    _hash(self):
        Generates a unique hash for the instance based on its class and `who_am_i` attribute.

    _cusiter(self):
        Returns an iterator for the object, enabling it to be used in loops.

    Notes
    -----
    - This metaclass assumes a manager class (`sys`) is available for interacting with the simulation.
    - Subclasses must implement the `set_object` method to define their specific setup behavior.
    """

    def __init__(cls, name, bases, class_dict):
        """
        Initializes the class and enforces required class-level attributes.

        Parameters
        ----------
        name : str
            The name of the class being created.
        bases : tuple
            Base classes of the new class.
        class_dict : dict
            Namespace of the new class.

        Raises
        ------
        NotImplementedError
            If a required attribute is missing or incorrectly typed.
        """
        super().__init__(name, bases, class_dict)
        # Assign class-level __iter__, __eq__, and __hash__ to make them work consistently
        cls.__iter__ = Simulation_Object._cusiter
        cls.__eq__ = Simulation_Object._eq
        cls.__hash__ = Simulation_Object._hash
        required_attributes = {
            "required_features": list,
            "numInstances": int,
            "part_types": PartDictSafe,
            "simulation_type": SinglePairDict,
            "config": ObjectConfigParams,
        }
        for attr, expected_type in required_attributes.items():
            if not hasattr(cls, attr):
                generic_type_exception(name, attr, expected_type)
            # Check if attribute is of the correct type
            elif not isinstance(getattr(cls, attr), expected_type):
                generic_type_exception(name, attr, expected_type)
    
    def __call__(cls, *args, **kwargs):
        """
        Creates a new instance and assigns default methods and attributes.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the instance.
        **kwargs : dict
            Keyword arguments for the instance.

        Returns
        -------
        object
            A new instance of the class.

        Raises
        ------
        NotImplementedError
            If a required instance attribute is missing or incorrectly typed.
        """
        # Create a new instance of the class
        instance = super().__call__(*args, **kwargs)
        # Assign the build_function to the class if it does not already have one

        helper_set_attribute(instance,"build_function",RoutineWithArgs())
        helper_set_attribute(instance,"set_object",Simulation_Object.set_object)
        helper_set_attribute(instance,"add_particle",Simulation_Object.add_particle)
        helper_set_attribute(instance,"get_owned_part",Simulation_Object.get_owned_part)
        helper_set_attribute(instance,"bond_owned_part_pair",Simulation_Object.bond_owned_part_pair)
        helper_set_attribute(instance,"change_part_type",Simulation_Object.change_part_type)
        helper_set_attribute(instance,"modify_system_attribute",generic_modify_system_attribute)
        helper_set_attribute(instance,"delete_owned_parts",Simulation_Object.delete_owned_parts)

        required_attributes = {"who_am_i": int,'type_part_dict'  :PartDictSafe,'associated_objects': list, "sys": espressomd.System}
        for attr, expected_type in required_attributes.items():
            # Check for required instance attribute `who_am_i`
            if not hasattr(instance, attr):
                generic_type_exception_inst(instance.__class__.__name__, attr, expected_type)
        return instance

    @staticmethod
    def _eq(self, other):
        """
        Compares two instances for equality based on their `who_am_i` attribute.

        Parameters
        ----------
        other : object
            The other instance to compare with.

        Returns
        -------
        bool
            True if the instances are equal, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False
        return getattr(self, "who_am_i", None) == getattr(other, "who_am_i", None)

    @staticmethod
    def _hash(self):
        """
        Generates a unique hash for the instance based on its class and `who_am_i` attribute.

        Returns
        -------
        int
            The hash value of the instance.
        """
        return hash((self.__class__, getattr(self, "who_am_i", None)))

    @staticmethod
    def _cusiter(self):
        """
        Returns an iterator for the object, enabling it to be used in loops.

        Returns
        -------
        iterator
            An iterator for the object.
        """
        """Return an iterator for the internal list, making the object iterable."""
        return iter([self])

    def set_object(self, *args,**kwargs):
        """
        Abstract method for setting up simulation objects. Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by a subclass.
        """
        logging.info(f"Self is: {self}")
        raise NotImplementedError("The 'set_object' method must be implemented in subclasses.")
    
    def delete_owned_parts(self):
        """
        Deletes all particles owned by the object and any associated objects.

        Iterates through `type_part_dict` to remove particles from the simulation.
        If `associated_objects` is defined, recursively deletes their owned particles.
        """
        if self.associated_objects!= None:
            for obj in self.associated_objects:
                obj.delete_owned_parts()
        for key,elem in self.type_part_dict.items():
            for prt in elem:
                prt.remove()

    def get_owned_part(self):
        """
        Returns a tuple:
        - tot_part: a flat list of all particle handles.
        - particle_chains: a list (same length as tot_part) where each element
                            is a list of (class_name, who_am_i) tuples representing
                            the chain of objects from 'self' to the object that directly
                            holds the particle.
        """
        tot_part = []
        particle_chains = []

        def worker(obj, chain):
            # Update the chain with the current object's identity.
            new_chain = chain + [(obj.__class__.__name__, obj.who_am_i)]
            # Get all the particle handles from the current object.
            parts = list(itertools.chain.from_iterable(obj.type_part_dict.values()))
            # For each particle, record both the particle and its chain.
            for part in parts:
                tot_part.append(part)
                particle_chains.append(new_chain)
            # Recurse into any associated objects.
            if obj.associated_objects is not None:
                for sub_obj in obj.associated_objects:
                    worker(sub_obj, new_chain)
                    
        worker(self, [])
        return tot_part, particle_chains
        
    def add_particle(self, type_name, pos, **kwargs):
        """
        Adds a particle to the simulation box.

        Ensures the particle type is declared in `part_types` and updates tracking structures.

        Parameters
        ----------
        type_name : str
            Name of the particle type.
        pos : iterable of shape (3,)
            Position of the particle in the simulation box.
        **kwargs : dict
            Additional properties for the particle.

        Returns
        -------
        ParticleHandle
            Handle to the added particle.

        Raises
        ------
        AssertionError
            If the particle type is not declared in `part_types`.
        """

        assert type_name in self.part_types.keys(), 'an object can only add partices of types declared in self.part_types'
        part_params={'type':self.part_types[type_name],'pos': pos}
        part_params.update(**kwargs)

        part_hndl = self.sys.part.add(**part_params)
        self.type_part_dict[type_name].append(part_hndl)
        self.modify_system_attribute(self,'part_types', lambda current_value: current_value.update({type_name: part_params['type']}))
        return part_hndl
    
    def bond_owned_part_pair(self,p1,p2, bond_handle=None):
        """
        Bonds the particle pair (p1,p2).

        Adds the bond to the system if not already present.

        Parameters
        ----------
        p1 : espressomd.particle.ParticleHandle
            The partner particle to bond with.
        p2 : espressomd.particle.ParticleHandle
            The partner particle to bond with.
        """
        if bond_handle is None:
            bond=self.params['bond_handle'].get_raw_handle()
        else:
            bond=bond_handle.get_raw_handle()

        if bond not in self.sys.bonded_inter:
            self.sys.bonded_inter.add(bond)
            logging.info(f'bond handle added to system for Object {self.__class__,self.who_am_i}')
        p1.add_bond((bond, p2))
    
    def change_part_type(self, particle, new_type_name):
        """
        Changes the type of an existing particle.

        Updates `type_part_dict` and modifies the particle's type in the simulation.

        Parameters
        ----------
        particle : ParticleHandle
            Handle to the particle being modified.
        new_type_name : str
            Name of the new particle type.

        Raises
        ------
        AssertionError
            If the new type is not declared in `part_types`.
        """
        assert new_type_name in self.part_types.keys(), 'an object can only add partices of types declared in self.part_types'
        current_key = next(key for key, values in self.type_part_dict.items() if particle in values)
        self.type_part_dict[current_key].remove(particle)
        particle.type=self.part_types[new_type_name]
        self.type_part_dict[new_type_name].append(particle)
        self.modify_system_attribute(self,'part_types', lambda current_value: current_value.update({new_type_name: self.part_types[new_type_name]}))


class ObjectConfigParams(dict):
    """
    Configuration parameters for simulation objects.

    The `ObjectConfigParams` class extends the dictionary to provide a structured way to manage
    configuration parameters for simulation objects. It includes validation and merging with
    class-level configurations.

    Attributes
    ----------
    common_keys : dict
        Common configuration keys with default values.

    Methods
    -------
    validate_and_join(class_config):
        Validates the current configuration against the class-level configuration and updates
        the instance with missing values from the class configuration.
    """
    common_keys = {'sigma': 1, 'espresso_handle': None, 'associated_objects': None, 'size': 1, 'n_parts': 1}

    def __init__(self, **kwargs):
        # Merge common_keys with kwargs; kwargs overwrite common_keys
        initial_data = {**self.common_keys, **kwargs}
        super().__init__(initial_data)
        self._allowed_keys = set(initial_data.keys())  # Track allowed keys


    def __setitem__(self, key, value):
        if key not in self._allowed_keys:
            raise KeyError(f"Cannot add new key '{key}' after initialization.")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        raise KeyError(f"Cannot delete key '{key}' from configuration.")

    def specify(self, **overrides):
        """
        Create a new instance configuration based on the class-level configuration,
        with specified overrides applied.

        Parameters
        ----------
        overrides : dict
            Key-value pairs to override in the configuration.

        Returns
        -------
        ObjectConfigParams
            A new ObjectConfigParams instance with the specified overrides.
        """
        # Ensure that the overrides only include allowed keys
        override_keys = set(overrides.keys())
        invalid_keys = override_keys - self._allowed_keys
        if invalid_keys:
            raise ValueError(f"Invalid keys in overrides: {invalid_keys}")

        # Create a new configuration with overrides applied
        new_config = ObjectConfigParams(**self)
        for key, value in overrides.items():
            new_config[key] = value
        return new_config
        
    def __repr__(self):
        """
        Returns a string representation of the ObjectConfigParams instance.

        Returns
        -------
        str
            String representation of the instance.
        """
        return f"ObjectConfigParams({super().__repr__()})"
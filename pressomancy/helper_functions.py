import numpy as np
from itertools import product
from collections import defaultdict
import inspect

class ManagedSimulation:
    """
    A decorator class to manage a singleton instance of a simulation object.

    The `ManagedSimulation` class enforces that only one instance of a decorated simulation class can exist at a time. It provides methods to initialize, reinitialize, and manage the instance, while maintaining a shared `espressomd.System` object. This class is especially useful for simulations where global state must be consistent across multiple components.

    Attributes
    ----------
    aClass : type
        The class being decorated and managed as a singleton.
    instance : object, optional
        The single instance of the decorated class. Initially set to `None`.
    init_args : tuple
        Arguments used during the initialization of the decorated class.
    init_kwargs : dict
        Keyword arguments used during the initialization of the decorated class.
    _espressomd_system : espressomd.System
        The shared ESPResSo system object, initialized during the first instantiation.
    __name__ : str
        The name of the singleton instance, including the decorated class name.
    __qualname__ : str
        The qualified name of the singleton instance, including the decorated class's qualified name.

    Methods
    -------
    __call__(*args, **kwargs):
        Creates and initializes the singleton instance, or raises an exception if it already exists.
    reinitialize_instance():
        Recreates the instance while preserving the shared ESPResSo system object and resets the system state.
    __getattr__(name):
        Forwards attribute access to the instance, raising an error if the instance is uninitialized.
    """

    def __init__(self, aClass):
        """
        Initializes the ManagedSimulation decorator.

        Parameters
        ----------
        aClass : type
            The class to be decorated and managed as a singleton.
        """
        self.aClass = aClass
        setattr(aClass, 'reinitialize_instance', self.reinitialize_instance)
        self.instance = None
        self.init_args = ()
        self.init_kwargs = {}
        self._espressomd_system = None  # Shared espressomd.System instance
        self.__name__ = f"Singleton({aClass.__name__})"
        self.__qualname__ = f"Singleton({aClass.__qualname__})"

    def __call__(self, *args, **kwargs):
        """
        Creates and initializes the singleton instance, or raises an exception if it already exists.

        If the instance does not exist, initializes the shared ESPResSo system object and the decorated class.
        If the instance already exists, raises a `SimulationExistsException`.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the decorated class constructor.
        **kwargs : dict
            Keyword arguments for the decorated class constructor. Includes optional `box_dim` to specify the
            simulation box dimensions.

        Returns
        -------
        ManagedSimulation
            The ManagedSimulation instance (not the decorated class instance).

        Raises
        ------
        SimulationExistsException
            If an instance of the decorated class already exists.
        """
        if self.instance is None:
            # Initialize the ESPResSo system object
            if self._espressomd_system is None:
                box_dim = kwargs.get('box_dim', [10, 10, 10])  # Default box dimensions
                self._espressomd_system = self.aClass._sys(box_l=box_dim)

            # Instantiate the decorated class and set its system attribute
            self.instance = self.aClass(*args, **kwargs)
            self.instance.sys = self._espressomd_system
            self.init_args = args
            self.init_kwargs = kwargs
        else:
            # Raise exception if a second instance is attempted
            frame = inspect.currentframe().f_back
            raise SimulationExistsException(
                f"An instance of {self.aClass.__name__} already exists at {frame.f_code.co_filename}, line {frame.f_lineno}"
            )
        return self  # Return the ManagedSimulation instance

    def reinitialize_instance(self):
        """
        Recreates the singleton instance without affecting the shared ESPResSo system object.

        This method resets the decorated class instance while preserving the ESPResSo system object.
        It clears particles, interactions, and thermostat settings in the system, ensuring a clean state.
        """
        if self.instance is not None:
            self.instance = self.aClass(*self.init_args, **self.init_kwargs)
            self.instance.sys = self._espressomd_system
            self.instance.sys.part.clear()
            self.instance.sys.non_bonded_inter.reset()
            self.instance.sys.bonded_inter.clear()
            self.instance.sys.thermostat.turn_off()

    def __getattr__(self, name):
        """
        Forwards attribute access to the singleton instance.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        object
            The requested attribute from the instance.

        Raises
        ------
        AttributeError
            If the singleton instance has not been initialized.
        """
        if self.instance is None:
            raise AttributeError(f"Instance of {self.aClass.__name__} has not been initialized.")
        return getattr(self.instance, name)

class SimulationExistsException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SinglePairDict(dict):
    """
    A dictionary wrapper that enforces unique key-value pairs across all instances.

    The `SinglePairDict` class ensures that each key and value are unique globally across all instances of the class.
    The dictionary is immutable after initialization, preventing modification or deletion of the stored key-value pair.

    Attributes
    ----------
    _global_registry : dict
        A class-level dictionary that tracks all key-value pairs globally across instances.

    Methods
    -------
    get_all_pairs():
        Returns a copy of the globally registered key-value pairs across all instances.

    Properties
    ----------
    key : object
        The single key stored in the dictionary.
    value : object
        The single value stored in the dictionary.

    Examples
    --------
    Creating a new `SinglePairDict`:
    >>> spd1 = SinglePairDict('key1', 'value1')
    >>> spd1.key
    'key1'
    >>> spd1.value
    'value1'

    Attempting to reuse an existing key or value:
    >>> spd2 = SinglePairDict('key1', 'value2')
    ValueError: Key 'key1' already exists in another instance.

    Retrieving all globally registered pairs:
    >>> SinglePairDict.get_all_pairs()
    {'key1': 'value1'}
    """

    # Class-level dictionary to track all unique key-value pairs across instances
    _global_registry = {}

    def __init__(self, key, value):
        """
        Initializes the dictionary with a single key-value pair.

        Parameters
        ----------
        key : object
            The key to store in the dictionary. Must be globally unique.
        value : object
            The value to store in the dictionary. Must be globally unique.

        Raises
        ------
        ValueError
            If the key or value already exists in another instance.
        """
        # Enforce global uniqueness for key and value
        if key in SinglePairDict._global_registry:
            raise ValueError(f"Key '{key}' already exists in another instance.")
        if value in SinglePairDict._global_registry.values():
            raise ValueError(f"Value '{value}' already exists in another instance.")

        # Initialize as a single-item dictionary
        super().__init__({key: value})

        # Register the key-value pair globally
        SinglePairDict._global_registry[key] = value

    def __setitem__(self, key, value):
        """
        Overrides the default method to prevent modification of the dictionary.

        Raises
        ------
        TypeError
            Always raised because item assignment is not supported.
        """
        raise TypeError("SinglePairDict does not support item assignment after initialization.")

    def __delitem__(self, key):
        """
        Overrides the default method to prevent deletion from the dictionary.

        Raises
        ------
        TypeError
            Always raised because item deletion is not supported.
        """
        raise TypeError("SinglePairDict does not support item deletion.")

    @property
    def key(self):
        """
        Retrieves the single key stored in the dictionary.

        Returns
        -------
        object
            The single key stored in the dictionary.
        """
        return next(iter(self.keys()))

    @property
    def value(self):
        """
        Retrieves the single value stored in the dictionary.

        Returns
        -------
        object
            The single value stored in the dictionary.
        """
        return next(iter(self.values()))

    @classmethod
    def get_all_pairs(cls):
        """
        Returns all globally registered key-value pairs across instances.

        Returns
        -------
        dict
            A copy of the globally registered key-value pairs.
        """
        return cls._global_registry.copy()

    def __repr__(self):
        """
        Returns a string representation of the dictionary.

        Returns
        -------
        str
            A string representation in the format `SinglePairDict(key: value)`.
        """
        return f"SinglePairDict({self.key!r}: {self.value!r})"
    
class PartDictSafe(dict):
    """
    A safe dictionary wrapper to enforce consistency and uniqueness of keys and values.

    `PartDictSafe` ensures that:
    - Keys are unique and cannot be reassigned once set.
    - Values are unique and cannot be associated with multiple keys.
    - A default value is provided for missing keys using a customizable default factory.

    This is especially useful for managing mappings where both the keys and values must remain consistent, 
    such as particle types and their properties in simulations.

    Attributes
    ----------
    default_factory : callable
        A function that returns the default value for missing keys. Defaults to `list`.

    Methods
    -------
    sanity_check(key, value):
        Validates that the key and value do not violate uniqueness constraints.
    set_default_factory(factory):
        Updates the default factory used to generate default values for missing keys.
    __setitem__(key, value):
        Sets a key-value pair in the dictionary after passing a sanity check.
    update(*args, **kwargs):
        Updates the dictionary with key-value pairs from another dictionary or iterable, enforcing sanity checks.
    __getitem__(key):
        Retrieves the value for a key, initializing it with the default value if the key does not exist.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the dictionary with optional initial data and a default factory.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the `dict` constructor.
        **kwargs : dict
            Keyword arguments passed to the `dict` constructor.
        """
        self.default_factory = list
        super().__init__(*args, **kwargs)

    def sanity_check(self, key, value):
        """
        Ensures the key and value do not violate uniqueness constraints.

        Parameters
        ----------
        key : object
            The key to validate.
        value : object
            The value to validate.

        Raises
        ------
        RuntimeError
            If the key already exists with a different value or the value is already associated with another key.
        """
        if value == self.default_factory():
            return
        current_value = self.get(key)
        if current_value == value:
            return

        if current_value is not None:
            raise RuntimeError(
                f"Key '{key}' already exists in `part_types` with a different value '{current_value}'. "
                f"Attempted to reset it to '{value}', which is not allowed."
            )

        if value in self.values():
            existing_key = next(k for k, v in self.items() if v == value)
            raise RuntimeError(
                f"Value '{value}' is already associated with key '{existing_key}' in `part_types`. "
                f"New entries must have unique values."
            )

    def __setitem__(self, key, value):
        """
        Sets a key-value pair in the dictionary after validating with a sanity check.

        Parameters
        ----------
        key : object
            The key to add or update.
        value : object
            The value to associate with the key.

        Raises
        ------
        RuntimeError
            If the key or value violates the uniqueness constraints.
        """
        self.sanity_check(key, value)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        """
        Updates the dictionary with key-value pairs from another dictionary or iterable.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the `dict.update` method.
        **kwargs : dict
            Keyword arguments passed to the `dict.update` method.

        Raises
        ------
        RuntimeError
            If any key-value pair violates the uniqueness constraints.
        """
        if args:
            iterable = args[0]
            for key, value in (iterable.items() if isinstance(iterable, dict) else iterable):
                self.sanity_check(key, value)
                super().__setitem__(key, value)

        for key, value in kwargs.items():
            self.sanity_check(key, value)
            super().__setitem__(key, value)

    def __getitem__(self, key):
        """
        Retrieves the value for a key, initializing it with the default value if the key does not exist.

        Parameters
        ----------
        key : object
            The key to retrieve.

        Returns
        -------
        object
            The value associated with the key or the default value if the key does not exist.
        """
        if key not in self:
            self[key] = self.default_factory()
        return super().__getitem__(key)

    def set_default_factory(self, factory):
        """
        Updates the default factory used to generate default values for missing keys.

        Parameters
        ----------
        factory : callable
            A function that returns the new default value for missing keys.
        """
        self.default_factory = factory

class RoutineWithArgs:
    """
    A wrapper class to manage callable routines with configurable arguments.

    The `RoutineWithArgs` class provides a way to encapsulate a callable function,
    allowing it to be called with predefined arguments. If no function is provided
    during initialization, a default routine (`generic_routine_per_volume`) is used.

    Attributes
    ----------
    func : callable
        The function to be called. Defaults to `generic_routine_per_volume`.
    num_monomers : int
        The number of monomers or items to process within the routine.

    Methods
    -------
    __call__(**kwargs)
        Invokes the encapsulated function with the provided keyword arguments.
    generic_routine_per_volume(**kwargs)
        A default routine to generate points within a spherical volume. Must be
        implemented by subclasses or overridden.
    """

    def __init__(self, func=None, num_monomers=1):
        """
        Initializes the RoutineWithArgs instance.

        Parameters
        ----------
        func : callable, optional
            The function to encapsulate. If not provided, `generic_routine_per_volume` is used.
        num_monomers : int, optional
            The number of monomers or items to process. Defaults to 1.
        """
        if func is None:
            self.func = self.generic_routine_per_volume
        else:
            self.func = func
        self.num_monomers = num_monomers

    def __call__(self, **kwargs):
        """
        Invokes the encapsulated function with the provided keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            The arguments to pass to the encapsulated function.

        Returns
        -------
        object
            The result of the encapsulated function call.
        """
        return self.func(**kwargs)
    @staticmethod
    def generic_routine_per_volume(**kwargs):
        """
        A placeholder for a default routine to generate points within a spherical volume.

        This method must be implemented by subclasses or overridden by specific instances.

        Parameters
        ----------
        **kwargs : dict
            The arguments required for the routine.

        Raises
        ------
        NotImplementedError
            If the method is called without being overridden.
        """
        raise NotImplementedError("Implement point generation method within a sphere.")

def load_coord_file(file_path):
    '''
    load coordinates from a text file. the function allways staples a (0,0,0) as the first row! 
    '''
    coordinates = np.zeros((1, 3), dtype=float)
    with open(file_path, mode='r') as source:
        for line in source:
            x, y, z = map(float, line.strip().split(','))
            coordinates = np.vstack([coordinates, [x, y, z]])
    return coordinates

def min_img_dist(s, t, box_dim):
    box_half = box_dim*0.5
    return np.remainder(s - t + box_half, box_dim) - box_half

def generate_random_unit_vectors(N_PART):
    z = np.random.uniform(-1, 1, N_PART)
    r = np.sqrt(1 - z*z)
    phi = np.random.uniform(0, 2*np.pi, N_PART)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y, z))

def get_neighbours(lattice_points: np.ndarray, volume_side: float, cuttoff: float=1.)-> defaultdict[int,list[float]]:
    '''
    returns grouped_indices, where grouped_indices is a dictionary that contains partile:neighours (within cuttoff)
    '''

    lattice_len = len(lattice_points)
    indices = np.arange(lattice_len)
    index_combinations = np.array(list(product(indices, repeat=2)))
    box_dim=np.ones(shape=3)*volume_side
    distances = np.linalg.norm(min_img_dist(lattice_points[index_combinations[:, 0]],
                                            lattice_points[index_combinations[:, 1]], box_dim=box_dim), axis=-1)
    valid_indices = index_combinations[(
        distances > 0) & (distances < cuttoff)]
    grouped_indices = defaultdict(list)

    for index_pair in valid_indices:
        grouped_indices[index_pair[0]].append(index_pair[1])
    return grouped_indices

def get_neighbours_cross_lattice(lattice1, lattice2, volume_side, cuttoff=1.):
    grouped_indices = defaultdict(list)
    points_a = np.array(lattice1)
    points_b = np.array(lattice2)
    num_b = len(points_b)
    indices_b = np.arange(num_b)
    box_dim=np.ones(3) * volume_side
    for id,point in enumerate(points_a):
        distances=np.linalg.norm(min_img_dist(point, points_b, box_dim=box_dim), axis=-1)
        mask=np.where(distances<cuttoff)
        grouped_indices[id]=list(indices_b[mask])
    
    return grouped_indices

def calculate_pairwise_distances(points_a, points_b, box_length=None):
    """
    Calculate the pairwise distances between two sets of points, considering 
    periodic boundary conditions if provided.

    Parameters
    ----------
    points_a : np.array of shape (N, 3)
        An array of points where N is the number of points in the first set.
    points_b : np.array of shape (M, 3)
        An array of points where M is the number of points in the second set.
    box_length : float, optional
        The length of the cubic box. If provided, periodic boundary conditions
        are applied.

    Returns
    -------
    distances : np.array of shape (N, M)
        A 2D array of pairwise distances between each point in `points_a` and 
        each point in `points_b`.
    """
    
    # Ensure inputs are numpy arrays
    points_a = np.array(points_a)
    points_b = np.array(points_b)
    
    # Get the number of points in each set
    num_a = len(points_a)
    num_b = len(points_b)
    
    # Create index combinations for pairwise comparisons
    indices_a = np.arange(num_a)
    indices_b = np.arange(num_b)
    
    # Create a grid of all pairwise combinations of indices
    index_combinations = np.array(list(product(indices_a, indices_b)))
    
    # Extract corresponding points for each pair
    point_pairs_a = points_a[index_combinations[:, 0]]  # Points from the first set
    point_pairs_b = points_b[index_combinations[:, 1]]  # Points from the second set
    
    # Calculate the minimum image distance with periodic boundary conditions
    distances = np.linalg.norm(min_img_dist(point_pairs_a, point_pairs_b, box_dim=np.ones(3) * box_length), axis=-1)
    # distances = np.linalg.norm(point_pairs_a-point_pairs_b, axis=-1)
    
    return distances

def fcc_lattice(radius, volume_side, scaling_factor=1.):
    '''
    Generate spherical volume positions centered on an fcc lattice, inside a cubic volume.
    The calculation requires the spherical volume radius and the side length of the cubic volume.
    returns lattice_points
    '''

    # lattice constant determined so that the spheres touch along the face diagonal of the unit lattice
    radius_scaled = radius*scaling_factor
    lattice_constant = 2 * radius_scaled / np.sqrt(2)
    num_points = int(np.ceil(volume_side / lattice_constant))
    # remove last row. num_points is rounded up, so he have to remove the last row to avoid overflowing the box (would end up creating compeltely overlapping particles in PBC)
    indices = np.arange(num_points-1)
    x, y, z = np.meshgrid(indices, indices, indices, indexing='ij')

    sum_indices = x + y + z
    mask = sum_indices % 2 == 0

    lattice_points = np.column_stack(
        (x[mask], y[mask], z[mask])) * lattice_constant + np.ones(shape=3)*radius
    # if np.isclose(min([x for x in calculate_pairwise_distances(lattice_points,lattice_points,box_length=volume_side) if x>0.01]),2 * radius_scaled):
    #     warnings.warn('box_l is not big enough to avoid pbc clipping of the partitioning!')
    return lattice_points

def make_centered_rand_orient_point_array(center=np.array([0,0,0]),sphere_radius=1.,num_monomers=1):
    shift = sphere_radius / num_monomers
    spacing_array = np.linspace(-sphere_radius,
                        sphere_radius, num_monomers + 1)[:-1] + shift
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x_points = center[0] + spacing_array * np.sin(phi) * np.cos(theta)
    y_points = center[1] + spacing_array * np.sin(phi) * np.sin(theta)
    z_points = center[2] + spacing_array * np.cos(phi)
    points = np.column_stack((x_points, y_points, z_points))
    direction_vector=points[-1]-points[0]
    orientation_vector = direction_vector / np.linalg.norm(direction_vector)
    return orientation_vector,points

def partition_cubic_volume(box_length, num_spheres, sphere_diameter, routine_per_volume=RoutineWithArgs(),flag='rand'):
    """
    Partition a cubic volume into spherical regions and generate points within 
    each spherical region using a routine provided by the user or a default routine.

    Parameters
    ----------
    box_length : float
        The length of the cubic box (assumed to be equal on all sides) in which 
        the spheres are placed.
    num_spheres : int
        The number of spherical regions to be placed within the cubic volume.
    sphere_diameter : float
        The diameter of each spherical region. The radius is calculated as half 
        of this value.
    routine_per_volume : RoutineWithArgs or callable, optional
        A function or callable object responsible for generating points within 
        each spherical volume. If not provided, a default `RoutineWithArgs` is used. 
        The callable is expected to accept the following parameters:
            - center: array-like, coordinates of the center of the sphere.
            - num_monomers: int, the number of points to generate within the sphere.
            - sphere_radius: float, the radius of the sphere.

    Returns
    -------
    sphere_centers : ndarray
        An array of shape `(num_spheres, 3)` representing the coordinates of the 
        centers of the spherical volumes.
    result : ndarray
        An array of shape `(num_spheres, num_monomers, 3)` representing the generated 
        points inside each spherical region. Each `num_monomers` corresponds to the 
        points inside one spherical volume.

    Raises
    ------
    AssertionError
        If the number of available spherical centers (volumes) is less than the number 
        of required spheres. This is addressed by dynamically scaling the lattice to 
        ensure enough space for all spheres.
    ValueError
        If `routine_per_volume.num_monomers` is not provided, or if it is zero or negative.

    Notes
    -----
    The function uses an FCC (face-centered cubic) lattice to place the centers of the 
    spheres within the cubic box. It dynamically adjusts the lattice scaling if the 
    initial placement does not provide enough spherical regions to meet `num_spheres`.

    The routine provided by `routine_per_volume` is responsible for generating the 
    internal structure (i.e., monomer points) within each spherical volume. The function 
    ensures that these points do not overlap with points in neighboring spheres by 
    applying periodic boundary conditions and checking the distances between points.

    Minimum image distance calculations are done using the `min_img_dist` function, which 
    ensures proper handling of periodic boundary conditions.
    """
    
    # Calculate the radius of each sphere
    sphere_radius = sphere_diameter * 0.5
    
    # Initialize variables for dynamic scaling of lattice
    volumes_to_fill = 0
    scaling = 1.0
    
    # Continue adjusting the lattice scaling until enough volumes (spheres) are available
    while volumes_to_fill < num_spheres:
        sphere_centers = fcc_lattice(radius=sphere_radius, volume_side=box_length, scaling_factor=scaling)
        volumes_to_fill = len(sphere_centers)
        print('num_spheres_needed, num_spheres_got ', num_spheres, volumes_to_fill)
        scaling -= 0.1
    print('scaling used: ', scaling + 0.1)
    # Randomly shuffle the available centers and select the required number of centers
    take_index = np.arange(len(sphere_centers))
    if flag=='rand':
        np.random.shuffle(take_index)
    take_index = take_index[:num_spheres]
    sphere_centers=sphere_centers[take_index]
    #grouped_volumes is a dictionary that contains all neighouring lattice sites sphere_diameter
    grouped_volumes=get_neighbours(sphere_centers,volume_side=box_length,cuttoff=sphere_diameter)

    # Ensure that we have enough spherical regions to accommodate the required number of spheres
    assert len(sphere_centers) >= num_spheres, 'Not enough volumes available. Consider introducing a scaling factor.'
    
    # Initialize an array to store the generated points inside each spherical region
    result = np.empty((num_spheres, routine_per_volume.num_monomers, 3))
    orientations=np.empty((num_spheres,3))

    # Perform the point generation routine if `num_monomers` not 0
    if routine_per_volume.num_monomers>1:
        grouped_positions = defaultdict(list)
        
        # For each selected center, generate points using the provided routine
        for i, center in enumerate(sphere_centers):
            valid_placement = False
            
            # Ensure the points do not overlap with points in neighboring volumes
            while not valid_placement:
                orientation, points = routine_per_volume(center=center, num_monomers=routine_per_volume.num_monomers, sphere_radius=sphere_radius)
                should_proceed = True
                
                # Check for overlaps with points in neighboring spheres
                for volume_id in grouped_volumes[i]:
                    if grouped_positions[volume_id]:
                        distances=calculate_pairwise_distances(points,grouped_positions[volume_id],box_length=box_length)
                        if np.any(distances < sphere_diameter / routine_per_volume.num_monomers):
                            should_proceed = False
                            break
                
                # If no overlaps were detected, finalize the placement of the points
                if should_proceed:
                    grouped_positions[i].extend(points)
                    result[i] = points
                    orientations[i] = orientation

                    valid_placement = True
    else:
        shpsd=sphere_centers
        result=shpsd
        orientations=generate_random_unit_vectors(len(sphere_centers))
        
    return sphere_centers, result, orientations

def partition_cubic_volume_oriented_rectangles(big_box_dim, num_spheres, small_box_dim, num_monomers):
    """
    Partition a cubic volume into smaller rectangular regions and generate oriented points within each region.

    This function divides a larger cubic box into smaller rectangular volumes based on the dimensions of the 
    smaller boxes provided. It then generates a specified number of points within each smaller volume, ensuring 
    they are oriented along a random direction.

    Parameters
    ----------
    big_box_dim : array-like of shape (3,)
        Dimensions of the larger cubic box (lengths along x, y, and z axes).
    num_spheres : int
        Number of smaller rectangular volumes to generate within the larger box.
    small_box_dim : array-like of shape (3,)
        Dimensions of the smaller boxes (lengths along x, y, and z axes).
    num_monomers : int
        Number of points to generate within each smaller box.

    Returns
    -------
    sphere_centers : ndarray of shape (num_spheres, 3)
        Coordinates of the centers of the selected rectangular volumes.
    result : ndarray of shape (num_spheres, num_monomers, 3)
        Generated points within each rectangular volume, oriented along a random direction.

    Raises
    ------
    AssertionError
        If the number of available rectangular volumes is less than `num_spheres`.

    Notes
    -----
    - The function uses the dimensions of `small_box_dim` to determine the number of partitions along each axis.
    - When there are fewer partitions along an axis (e.g., one partition), alternate boxes along that axis are 
      adjusted to ensure even distribution.
    - The generated points within each smaller box are spaced along a single direction determined by a random angle.

    Examples
    --------
    Partition a 10x10x10 box into smaller 2x2x2 volumes and generate 5 points in each volume:
    >>> big_box_dim = np.array([10.0, 10.0, 10.0])
    >>> small_box_dim = np.array([2.0, 2.0, 2.0])
    >>> num_spheres = 10
    >>> num_monomers = 5
    >>> centers, points = partition_cubic_volume_oriented_rectangles(big_box_dim, num_spheres, small_box_dim, num_monomers)
    """
    _, _, sphere_diameter = small_box_dim
    sphere_radius = sphere_diameter * 0.5

    x_partitions, y_partitions, z_partitions = (
        big_box_dim // small_box_dim).astype(int)

    x_len, y_len, z_len = small_box_dim
    x_coords = np.linspace(
        0.5 * x_len, big_box_dim[0] - 0.5 * x_len, x_partitions)
    y_coords = np.linspace(
        0.5 * y_len, big_box_dim[1] - 0.5 * y_len, y_partitions)
    z_coords = np.linspace(
        0.5 * z_len, big_box_dim[2] - 0.5 * z_len, z_partitions)

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    sphere_centers = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Adjust coordinates for partitions equal to 1
    if x_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 0] = big_box_dim[0] - 0.5 * x_len

    if y_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 1] = big_box_dim[1] - 0.5 * y_len

    if z_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 2] = big_box_dim[2] - 0.5 * z_len

    assert len(sphere_centers) >= num_spheres, \
        'Must be enough possible volumes. Introduce a scaling factor.'

    take_index = np.arange(len(sphere_centers))
    np.random.shuffle(take_index)
    take_index = take_index[:num_spheres]
    shift = sphere_radius / num_monomers
    alphas = np.linspace(-sphere_radius,
                         sphere_radius, num_monomers + 1)[:-1] + shift
    result = np.empty((num_spheres, num_monomers, 3))
    for i, iid in enumerate(take_index):
        center = sphere_centers[iid]
        theta = np.random.uniform(0, 2 * np.pi)
        phi = 0.
        x_points = center[0] + alphas * np.sin(phi) * np.cos(theta)
        y_points = center[1] + alphas * np.sin(phi) * np.sin(theta)
        z_points = center[2] + alphas * np.cos(phi)
        result[i] = np.column_stack((x_points, y_points, z_points))

    return sphere_centers[take_index], result

def generate_positions(no_objects, box_l, min_distance):
    quadriplex_positions = []
    while len(quadriplex_positions) < no_objects:
        center = box_l/2.
        factor = 1-min_distance/box_l
        new_position = center + factor*box_l*(np.random.random(3) - 0.5)
        # new_position = np.random.random(3) * box_l
        if all(np.linalg.norm(new_position - existing_position) >= min_distance
                for existing_position in quadriplex_positions):
            quadriplex_positions.append(new_position)

    return np.array(quadriplex_positions)

def generate_positions_directed_triples(no_objects, box_l, min_distance, director_list):
    assert len(
        director_list) == no_objects // 3, "Length of directorors must be one-third of no_objects"
    quadriplex_positions = []
    index = 0
    while len(quadriplex_positions) < no_objects:
        center = box_l/2.
        factor = 1-min_distance/box_l
        new_position = center + factor*box_l*(np.random.random(3) - 0.5)
        if all(np.linalg.norm(new_position - existing_position) >= min_distance
                for existing_position in quadriplex_positions):
            quadriplex_positions.append(new_position)
            quadriplex_positions.append(new_position+2.*director_list[index])
            quadriplex_positions.append(new_position-2.*director_list[index])
            index += 1
    return np.array(quadriplex_positions)

def get_orientation_vec(pos):
    '''
    Calculates the principal gyration axis of a filement as the orientation of a filament. Sometimes the np.linalg.eig() returns a complex number with 0 complex part wich confuses espresso. Therefore the ret values is cast to float explicitly

    :return: float | normalised principal gyration axis

    '''
    dip_3d = np.array(pos)
    r_cm = np.mean(dip_3d, axis=0)
    gyration_tensor_xx = np.mean(
        [(x-r_cm[0])*(x-r_cm[0]) for (x, y, z) in dip_3d])
    gyration_tensor_yy = np.mean(
        [(y-r_cm[1])*(y-r_cm[1]) for (x, y, z) in dip_3d])
    gyration_tensor_zz = np.mean(
        [(z-r_cm[2])*(z-r_cm[2]) for (x, y, z) in dip_3d])
    gyration_tensor_xy = np.mean(
        [(x-r_cm[0])*(y-r_cm[1]) for (x, y, z) in dip_3d])
    gyration_tensor_xz = np.mean(
        [(x-r_cm[0])*(z-r_cm[2]) for (x, y, z) in dip_3d])
    gyration_tensor_yz = np.mean(
        [(y-r_cm[1])*(z-r_cm[2]) for (x, y, z) in dip_3d])
    gyration_tensor_element = [[gyration_tensor_xx, gyration_tensor_xy, gyration_tensor_xz],
                               [gyration_tensor_xy, gyration_tensor_yy,
                                   gyration_tensor_yz],
                               [gyration_tensor_xz, gyration_tensor_yz, gyration_tensor_zz]]
    res, egiv = np.linalg.eig(gyration_tensor_element)
    pr_comp = egiv[:, np.argmax(res)]
    pr_comp /= np.linalg.norm(pr_comp)
    return np.array(pr_comp, float)

def get_cross_lattice_noninterceting_volumes(sphere_centers_long, sph_diam_log,sphere_centers_short, grouped_part_pos_short,sph_diam_short,box_len):
    # assume lattice doesn need to use scaling and it fits in the box for now!!!

    neigh=get_neighbours_cross_lattice(sphere_centers_long,sphere_centers_short,
    box_len,cuttoff=(sph_diam_log+sph_diam_short)*0.5)
    aranged_cross_lattice_options={}
    for vol_id,associated_vol_ids in neigh.items():
        mask=[]
        if associated_vol_ids:
            for as_vol_id in associated_vol_ids:
                res=calculate_pairwise_distances([sphere_centers_long[vol_id]], grouped_part_pos_short[as_vol_id], box_length=box_len)
                mask.append(all([x>0.5*sph_diam_short for x in res if not np.isclose(x,0.)])) 
        aranged_cross_lattice_options[vol_id]=mask
    return aranged_cross_lattice_options
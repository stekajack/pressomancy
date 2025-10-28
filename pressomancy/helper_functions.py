import numpy as np
from itertools import product
from collections import defaultdict
import inspect
import logging
import warnings

class MissingFeature(Exception):
    pass

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
        It clears particles, interactions, constraints, actors, and thermostat settings in the system, ensuring a clean state.

        note: it is much faster without reseting non_bonded_interactions. You can do this manually, or you can uncomment the temporary fix for this.
        """
        if self.instance is not None:
            self.instance = self.aClass(*self.init_args, **self.init_kwargs)
            self.instance.sys = self._espressomd_system
            self.instance.sys.part.clear()
            # self.instance.sys.non_bonded_inter.reset() #this method is not working properly
            # self.instance.reset_non_bonded_inter() # temporary fix (uncomment)
            self.instance.sys.bonded_inter.clear()
            self.instance.sys.thermostat.turn_off()
            self.instance.sys.constraints.clear()
            # self.instance.sys.actors.clear()
            self.instance.sys.magnetostatics.clear()

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
    
    def __dir__(self):
        """
        Returns the list of attributes for the singleton instance and the ManagedSimulation class.

        This method combines the attributes of the singleton instance (if initialized) and the class itself,
        allowing for introspection of both the wrapped object's methods and the management-related methods
        provided by the decorator.

        Returns
        -------
        list
            A list of attribute names for the singleton instance and the ManagedSimulation class, including
            all attributes of both.

        Notes
        -----
        - This method is useful for introspection or debugging, providing a full list of methods and attributes
        available for the object managed by the decorator.
        - The singleton instance's attributes will be included in the result if it has been initialized; otherwise,
        only the `ManagedSimulation` class attributes will be returned.
        """
        if self.instance is not None:
            return dir(self.instance)  # Return attributes of the original class
        return dir(type(self))  # Return attributes of the decorator itself


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

    def key_for(self, value):
        """
        Return the (unique) key for `value`.

        Raises:
            KeyError:   if no key maps to `value`.
        """
        rek_keys=[]
        for val in np.atleast_1d(value):
            for k, v in self.items():
                if v == val:
                    rek_keys.append(k)
            if not rek_keys:
                raise KeyError(f"No key found for value {val}")

        return rek_keys

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

    def __init__(self, func=None, num_monomers=1, monomer_size=1., spacing=None):
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
        self.spacing = spacing
        self.monomer_size = monomer_size

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

def fold_coords(points, box_dim):
    return np.mod(points, box_dim)

def min_img_dist(s, t, box_dim):
    """
    Compute minimum image distance between s and t under periodic boundary conditions.

    Parameters
    ----------
    s : iterable of float, shape (..., 3)
        Source points.
    t : iterable of float, shape (..., 3)
        Target points.
    box_dim : interable of float, shape (3,)
        Cuboid box dimensions [Lx, Ly, Lz].

    Returns
    -------
    np.ndarray
        Minimum image displacement vectors.
    """
    box_dim = np.asarray(box_dim)
    box_half = box_dim*0.5
    return np.remainder(s - t + box_half, box_dim) - box_half

def generate_random_unit_vectors(N_PART):
    z = np.random.uniform(-1, 1, N_PART)
    r = np.sqrt(1 - z*z)
    phi = np.random.uniform(0, 2*np.pi, N_PART)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y, z))

def normalize_vectors(array_of_vectors, axis=-1):
    array_of_vectors= np.asarray(array_of_vectors)
    norms_array = np.atleast_1d(np.linalg.norm(array_of_vectors, axis=axis))
    norms_array[norms_array==0] = 1
    return array_of_vectors / np.expand_dims(norms_array, axis)

def build_grid_and_adjacent(lattice_points, volume_side, cell_size):
    """
    Builds a grid dictionary mapping each cell id (tuple) to a list of particle indices, and
    an adjacent-cells dictionary mapping each occupied cell id to a list of its adjacent cell ids (including itself),
    taking periodic boundary conditions into account.

    Parameters
    ----------
    lattice_points : np.ndarray of shape (N, 3)
        Array of particle positions.
    volume_side : float
        The side length of the cubic volume.
    cell_size : float
        The grid cell size (typically set equal to the cuttoff distance).

    Returns
    -------
    grid : defaultdict(list)
        Dictionary mapping cell id (tuple of ints) to a list of particle indices in that cell.
    num_cells : int
        The number of cells per dimension.
    adjacent : dict
        Dictionary mapping each occupied cell id to a list of adjacent cell ids (as tuples), with periodic boundaries.
    """
    num_cells = int(np.ceil(volume_side / cell_size))
    # Compute cell indices for all points in one go using vectorized operations.
    cells = np.floor(lattice_points / cell_size).astype(int) % num_cells
    grid = defaultdict(list)
    # Group indices by cell ID.
    for idx, cell in enumerate(cells):
        grid[tuple(cell)].append(idx)
    # Precompute neighbor offsets (all combinations of -1, 0, 1 in 3 dimensions).
    neighbor_offsets = list(product([-1, 0, 1], repeat=3))
    
    # Build the adjacent cells dictionary only for the occupied cells.
    adjacent = {}
    for cell in grid.keys():
        cell_arr = np.array(cell)
        # For each offset, compute the neighboring cell id with periodic wrapping.
        adjacent[cell] = [tuple((cell_arr + np.array(offset)) % num_cells) for offset in neighbor_offsets]
    
    return grid, adjacent

def get_neighbours(lattice_points: np.ndarray, volume_side: float, cuttoff: float = 1., map_indices=None) -> defaultdict:
    """
    Returns grouped_indices, where grouped_indices is a dictionary that maps each particle index
    to a list of neighbor indices within the cuttoff distance. Uses a grid-based method for efficiency,
    and reuses the min_img_dist function for distance calculations.
    
    Parameters
    ----------
    lattice_points : np.ndarray of shape (N, 3)
        Array of particle positions.
    volume_side : float
        The side length of the cubic volume.
    cuttoff : float, optional
        The neighbor distance threshold.
    
    Returns
    -------
    grouped_indices : defaultdict[int, list[int]]
        Dictionary mapping each particle index to a list of neighbor indices.
    """
    # map indices (optional)
    if map_indices is None:
        map_indices = [i for i in range(len(lattice_points))]

    # Use cuttoff as the grid cell size.
    cell_size = cuttoff
    grid, adjacent_cells = build_grid_and_adjacent(lattice_points, volume_side, cell_size)
    
    grouped_indices = defaultdict(list)
    box_dim = np.ones(3) * volume_side
    
    # For each occupied cell in the grid...
    for cell, indices in grid.items():
        # Get the list of adjacent cells (neighbors) for this cell.
        neighbor_cells = adjacent_cells[cell]
        # For every particle in the current cell...
        for i in indices:
            # Check particles in each neighboring cell.
            for adj_cell in neighbor_cells:
                for j in grid.get(adj_cell, []):
                    if j == i:
                        continue
                    # Use min_img_dist to compute the distance with periodic boundaries.
                    diff = min_img_dist(lattice_points[i], lattice_points[j], box_dim=box_dim)
                    if np.linalg.norm(diff) <= cuttoff:
                        grouped_indices[i].append(j)

    return grouped_indices

def get_neighbours_ordered(lattice_points: np.ndarray, volume_side: float, cuttoff: float = 1., map_indices=None, periodicity=[True,True,True]) -> defaultdict:
    """
    Returns grouped_indices, where grouped_indices is a dictionary that maps each particle index
    to a list of neighbor indices within the cuttoff distance. Uses a grid-based method for efficiency,
    and reuses the min_img_dist function for distance calculations.
    
    Parameters
    ----------
    lattice_points : np.ndarray of shape (N, 3)
        Array of particle positions.
    volume_side : float
        The side length of the cubic volume.
    cuttoff : float, optional
        The neighbor distance threshold.
    
    Returns
    -------
    grouped_indices : defaultdict[int, list[int]]
        Dictionary mapping each particle index to a list of neighbor indices.

    note:
        - if no index mapping is provided, particle indices will be assumed to start on 0 and end on n_particles-1
    """
    # map indices (optional)
    if map_indices is None:
        map_indices = [i for i in range(len(lattice_points))]
    # Use cuttoff as the grid cell size.
    cell_size = cuttoff
    grid, adjacent_cells = build_grid_and_adjacent(lattice_points, volume_side, cell_size)
    
    grouped_indices = defaultdict(list)
    box_dim = np.ones(3) * volume_side

    for i, p in enumerate(periodicity):
        if not p:
            box_dim[i] = 1000000
    
    # For each occupied cell in the grid...
    for cell, indices in grid.items():
        # Get the list of adjacent cells (neighbors) for this cell.
        neighbor_cells = adjacent_cells[cell]
        # For every particle in the current cell...
        for i in indices:
            grouped_indices_dist = []
            # Check particles in each neighboring cell.
            for adj_cell in neighbor_cells:
                for j in grid.get(adj_cell, []):
                    if j == i:
                        continue
                    # Use min_img_dist to compute the distance with periodic boundaries.
                    diff = min_img_dist(lattice_points[i], lattice_points[j], box_dim=box_dim)
                    dist = np.linalg.norm(diff)
                    if dist <= cuttoff:
                        grouped_indices_dist.append((j, dist))
            grouped_indices_dist = list(set(id_ for id_, _ in sorted(grouped_indices_dist, key=lambda x: x[1])))
            grouped_indices[map_indices[i]] = [map_indices[j] for j in grouped_indices_dist]
    return grouped_indices

def get_neighbours_cross_lattice(lattice1, lattice2, box_lengths, cuttoff=1.):
    """
    Get neighbors between two lattices in a cuboid under PBC using minimum image convention.

    Parameters
    ----------
    lattice1 : np.ndarray, shape (N1, 3)
    lattice2 : np.ndarray, shape (N2, 3)
    box_len : array-like of shape (3,)
    cutoff : float

    Returns
    -------
    dict
        {index in lattice1: [indices in lattice2 within cutoff]}
    """
    if isinstance(box_lengths, float):
        box_lengths = box_lengths * np.ones(3)
    box_lengths = np.asarray(box_lengths)
    grouped_indices = defaultdict(list)
    points_a = np.atleast_2d(lattice1)
    points_b = np.atleast_2d(lattice2)
    num_b = len(points_b)
    indices_b = np.arange(num_b)
    for id,point in enumerate(points_a):
        distances=np.linalg.norm(min_img_dist(point, points_b, box_dim=box_lengths), axis=-1)
        mask=np.where(distances<=cuttoff)
        grouped_indices[id]=list(indices_b[mask])
    
    return grouped_indices

def calculate_pair_distances(points_a, points_b, box_lengths):
    """
    Calculate the pairwise distances between two sets of points under periodic boundary conditions.

    Parameters
    ----------
    points_a : np.ndarray, shape (N, 3)
        First set of 3D points.
    points_b : np.ndarray, shape (M, 3)
        Second set of 3D points.
    box_length : float or array-like of shape (3,), optional
        Applies periodic boundary conditions for a cubic or cuboid box.

    Returns
    -------
    distances : np.ndarray, shape (N*M,)
        1D array of distances between all pairs (a_i, b_j). dist(i,j) = distances[i*M + j]
    """
    if isinstance(box_lengths, float):
        box_lengths = box_lengths * np.ones(3)
    box_lengths = np.asarray(box_lengths)

    # Ensure inputs are numpy arrays
    points_a = np.atleast_2d(points_a)
    points_b = np.atleast_2d(points_b)
    
    # Get the number of points in each set
    num_a = len(points_a)
    num_b = len(points_b)
    
    # Create index combinations for pair comparisons
    indices_a = np.arange(num_a)
    indices_b = np.arange(num_b)
    
    # Create a grid of all pair combinations of indices
    index_combinations = np.array(list(product(indices_a, indices_b)))
    
    # Extract corresponding points for each pair
    point_pairs_a = points_a[index_combinations[:, 0]]  # Points from the first set
    point_pairs_b = points_b[index_combinations[:, 1]]  # Points from the second set
    
    # Calculate the minimum image distance with periodic boundary conditions
    distances = np.linalg.norm(min_img_dist(point_pairs_a, point_pairs_b, box_dim=box_lengths), axis=-1)

    # displacements = np.linalg.norm(min_img_dist(points_a[:, None, :], points_b[None, :, :], box_dim), axis=-1)
    
    return distances

def fcc_lattice(radius, volume_sides, scaling_factor=1., max_points_per_side=100):
    """
    Generates a face-centered cubic (FCC) lattice of points within a cuboid volume. The function creates an FCC crystal structure where spheres of given radius are arranged such that they touch along the face diagonal of the unit lattice.

    Parameters
    ----------
    radius : float
        Radius of the spheres in the lattice.
    volume_side : iterable of float of size 3
        Length of the cuboid volume's sides.
    scaling_factor : float, optional
        Factor to scale the radius of the spheres. Default is 1.0.
    max_points_per_side : int, optional
        Maximum number of points allowed per dimension. Default is 100.
        If exceeded, lattice constant is increased.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing the coordinates of the lattice points,
        where N is the number of points in the FCC lattice.

    Notes
    -----
    - The lattice constant is calculated as 2*radius_scaled/sqrt(2), where 
      radius_scaled = radius*scaling_factor.
    - If the number of points per side exceeds max_points_per_side, the lattice
      constant is gradually increased until the constraint is satisfied.
    - The function ensures the lattice fits within the given volume by removing 
      the last row of points to avoid periodic boundary condition overlaps.
    - When the lattice constant is increased, a warning message is logged with 
      the new value.
    """
    assert len(volume_sides)==3, "this metthod assumes volume_sides to be have len of 3"
    volume_sides = np.asarray(volume_sides)

    radius_scaled = radius*scaling_factor
    lattice_constant = 2 * radius_scaled / np.sqrt(2)
    while True:
        num_points = np.ceil(volume_sides / lattice_constant).astype(int)
        if (num_points <= max_points_per_side).any():
            break
        lattice_constant *= 1.1
        logging.info('lattice_constant increased to %s becaouse %s bigger than %s', lattice_constant,num_points,max_points_per_side)
   
    indices = [np.arange(num-1) for num in num_points ]
    x, y, z = np.meshgrid(indices[0], indices[1], indices[2], indexing='ij')
    sum_indices = x + y + z
    mask = sum_indices % 2 == 0
    lattice_points = np.column_stack(
        (x[mask], y[mask], z[mask])) * lattice_constant + np.ones(shape=3)*radius
    # if np.isclose(min([x for x in calculate_pair_distances(lattice_points,lattice_points,box_length=volume_side) if x>0.01]),2 * radius_scaled):
    #     warnings.warn('box_l is not big enough to avoid pbc clipping of the partitioning!')
    return lattice_points

def make_centered_rand_orient_point_array(center=np.array([0,0,0]), sphere_radius=1., num_monomers=1, spacing=None, box_lengths=None):
    """
    Creates an array of points centered at a given position with random orientation.This function generates a linear array of points in 3D space, centered at a specified position with random orientation. It also returns the normalized orientation vector of the array.

    Parameters
    ----------
    center : numpy.ndarray, default=np.array([0,0,0])
        The center point of the array in 3D space (x,y,z coordinates)
    sphere_radius : float, default=1.0
        The radius of the sphere containing the points
    num_monomers : int, default=1
        The number of points to generate
    spacing : float, optional
        If provided, sets fixed spacing between points. The total chain length will be spacing * (num_monomers - 1), and the points will be centered around center.
    Returns
    -------
    tuple
        A tuple containing:
        - orientation_vector (numpy.ndarray): Normalized vector indicating array orientation
        - points (numpy.ndarray): Array of 3D coordinates for each point
    Notes
    -----
    When spacing is provided, the positions along the line are given by:
    
        positions = spacing * (np.arange(num_monomers) - (num_monomers - 1)/2)
    
    ensuring that the distance between consecutive points is exactly 'spacing' and that the center of mass is at 0.
    The points are then rotated by a random orientation (given by theta and phi) and shifted by 'center'.
    """
    
    if spacing is not None:
        positions = spacing * (np.arange(num_monomers) - (num_monomers - 1) / 2)
    else:
        shift = sphere_radius / num_monomers
        positions = np.linspace(-sphere_radius,
                        sphere_radius, num_monomers + 1)[:-1] + shift
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x_points = center[0] + positions * np.sin(phi) * np.cos(theta)
    y_points = center[1] + positions * np.sin(phi) * np.sin(theta)
    z_points = center[2] + positions * np.cos(phi)
    points = np.column_stack((x_points, y_points, z_points))
    direction_vector=points[-1]-points[0]
    orientation_vector = direction_vector / np.linalg.norm(direction_vector)
    orientation_vectors = np.broadcast_to(orientation_vector, points.shape).copy()
    return orientation_vectors,points

def partition_cuboid_volume(box_lengths, num_spheres, sphere_diameter, routine_per_volume=RoutineWithArgs(), flag='rand'):
    """
    Partitions a cuboid volume into spherical regions and generates points within them.
    This function creates a face-centered cubic (FCC) lattice of spheres within a cubic volume; and optionally, generates points within each sphere according to a specified routine.
    
    Parameters
    ----------
    box_lengths : iterable of float of len 3
        The length of the cuboid volume's sides.
    num_spheres : int
        The desired number of spherical regions to create.
    sphere_diameter : float
        The diameter of each spherical region.
    routine_per_volume : RoutineWithArgs, optional
        A callable object that generates points within each sphere. Default is empty RoutineWithArgs.
    flag : str, optional
        Determines the arrangement of sphere centers. 'rand' for random shuffling. Default is 'rand'.
    
    Returns
    -------
    list of tuples
        Each tuple contains:
        - center (array-like): The coordinates of the sphere's center
        - points (array-like): The generated points within the sphere (or center if no routine)
        - orientation (array-like): The orientation vector for the sphere
    """
    box_lengths= np.asarray(box_lengths)
    sphere_radius = sphere_diameter * 0.5    
    scaling = 1.0
    
    # Adjust scaling until we have enough sphere centers
    while True:
        sphere_centers = fcc_lattice(radius=sphere_radius, volume_sides=box_lengths, scaling_factor=scaling)
        volumes_to_fill=len(sphere_centers)
        logging.info('num_spheres_needed, num_spheres_got: %s', (num_spheres, volumes_to_fill))
        if  volumes_to_fill>= num_spheres:
            break
        scaling -= 0.1
    logging.info('scaling used: %s', scaling)

    # Center point distribution in box
    min_centers = np.min(sphere_centers, axis=0)
    max_centers = np.max(sphere_centers, axis=0)
    sphere_centers += box_lengths/2 - (min_centers + max_centers)/2

    # Randomly shuffle the available centers and select the required number of centers
    take_index = np.arange(len(sphere_centers))
    if flag=='rand':
        np.random.shuffle(take_index)
    take_index = take_index[:num_spheres]
    sphere_centers=sphere_centers[take_index]  
    # Initialize an array to store the generated points inside each spherical region
    results = [None] * num_spheres
    orientations = [None] * num_spheres
    # Perform the point generation routine if `num_monomers` not 0
    if routine_per_volume.num_monomers>1:
        if np.all(box_lengths==box_lengths[0]):
            warnings.warn("this methods assumes cubic system box for num_monomers > 1")
        box_length = box_lengths[0]
        grouped_positions = defaultdict(list)
        #grouped_volumes is a dictionary that contains all neighouring lattice sites sphere_diameter 
        grouped_volumes=get_neighbours(sphere_centers,volume_side=box_length,cuttoff=sphere_diameter)
        for i, center in enumerate(sphere_centers):
            valid_placement = False
            while not valid_placement:
                orientations, points = routine_per_volume(
                    center=center, num_monomers=routine_per_volume.num_monomers, sphere_radius=sphere_radius, spacing=routine_per_volume.spacing,
                    box_lengths=box_lengths)
                should_proceed = True
                
                # Check for overlaps with points in neighboring spheres
                for volume_id in grouped_volumes[i]:
                    if grouped_positions[volume_id]:
                        distances = calculate_pair_distances(points, grouped_positions[volume_id], box_lengths=box_length)
                        if np.any(distances <= routine_per_volume.monomer_size):
                            should_proceed = False
                            break
                
                if should_proceed:
                    grouped_positions[i].extend(points)
                    results[i] = points
                    res_orientations[i] = orientations
                    valid_placement = True
    else:
        results=sphere_centers
        res_orientations=generate_random_unit_vectors(len(sphere_centers))
    return sphere_centers, results, res_orientations

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
    >>> centers, points = partition_cuboid_volume_oriented_rectangles(big_box_dim, num_spheres, small_box_dim, num_monomers)
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

def generate_positions(self, min_distance):
    """
    Generates random positions for objects in the simulation box, ensuring minimum distance between positions. Completely naive implementation

    :param min_distance: float | The minimum allowed distance between objects.
    :return: np.ndarray | Array of generated positions.
    """
    object_positions = []
    while len(object_positions) < self.no_objects:
        new_position = np.random.random(3) * self.sys.box_l
        if all(np.linalg.norm(new_position - pos) >= min_distance for pos in self.sys.part.all().pos):
            if all(np.linalg.norm(new_position - existing_position) >= min_distance for existing_position in object_positions):
                object_positions.append(new_position)
        logging.info(f'position casing progress: {len(object_positions)/self.no_objects}')

    return np.array(object_positions)

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

def get_cross_lattice_nonintersecting_volumes(current_lattice_centers, current_lattice_grouped_part_pos, current_lattice_diam,other_lattice_centers, other_lattice_grouped_part_pos,other_lattice_diam,box_lengths, mode='cross_volumes'):
    """
    Calculate non-intersecting volumes between particles in two different lattices. This function determines which volumes from one lattice do not intersect with volumes from another lattice,
    considering periodic boundary conditions.

    Parameters
    ----------
    current_lattice_centers : array-like
        Centers of volumes in the first lattice.
    current_lattice_grouped_part_pos : array-like 
        Particle positions grouped by volume for the first lattice.
    current_lattice_diam : float
        Diameter of particles in the first lattice.
    other_lattice_centers : array-like
        Centers of volumes in the second lattice.
    other_lattice_grouped_part_pos : array-like
        Particle positions grouped by volume for the second lattice.
    other_lattice_diam : float
        Diameter of particles in the second lattice.
    box_lengths : float
        Length of the periodic box.
    mode : str, optional
        Mode of calculation, either 'cross_parts' or 'cross_volumes'. Default is 'cross_volumes'.

    Returns
    -------
    dict
        Dictionary with volume IDs as keys and lists of boolean masks as values.
        Each mask indicates whether the volume from the first lattice intersects
        with corresponding volumes from the second lattice.

    Notes
    -----
    The function uses a cutoff distance of (d1 + d2)/2 where d1, d2 are the 
    diameters of particles in respective lattices. Particle pairs are considered
    non-intersecting if their separation is greater than (d1/n1 + d2/n2)/2,
    where n1, n2 are the number of particles in respective volumes.
    """
    
    neigh=get_neighbours_cross_lattice(current_lattice_centers,other_lattice_centers,
    box_lengths, cuttoff=(current_lattice_diam+other_lattice_diam)*0.5)
    aranged_cross_lattice_options={}
    if mode=='cross_parts':
        fact=pow(2,1/6)
        new_crit=((current_lattice_diam/current_lattice_grouped_part_pos.shape[1])*fact+(other_lattice_diam/other_lattice_grouped_part_pos.shape[1])*fact)*0.5
        current_lattice_dat=current_lattice_grouped_part_pos
        other_lattice_dat=other_lattice_grouped_part_pos

    elif mode=='cross_volumes':
        new_crit=(current_lattice_diam+other_lattice_diam)*0.5
        current_lattice_dat=current_lattice_centers
        other_lattice_dat=other_lattice_centers
    else:
        raise ValueError('mode must be either cross_parts or cross_volumes')
    for vol_id,associated_vol_ids in neigh.items():
        mask=[]
        if associated_vol_ids:
            for as_vol_id in associated_vol_ids:
                res=calculate_pair_distances(current_lattice_dat[vol_id], other_lattice_dat[as_vol_id], box_lengths=box_lengths)
                mask.append(all([x>=new_crit for x in res if not np.isclose(x,0.)])) 
        aranged_cross_lattice_options[vol_id]=mask
    return aranged_cross_lattice_options

def align_vectors(v1, v2):
    """
    Compute the rotation matrix that aligns vector v1 to vector v2.

    Args:
        v1 (numpy.ndarray): The initial vector to align.
        v2 (numpy.ndarray): The target vector to align with.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix that aligns v1 with v2.

    The function handles special cases where the vectors are already aligned or are opposite.
    It uses Rodrigues' rotation formula for general cases.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross_prod = np.cross(v1, v2)
    sin_theta = np.linalg.norm(cross_prod)
    cos_theta = np.dot(v1, v2)
    if np.isclose(cos_theta, 1.0):
        return np.eye(3)
    if np.isclose(cos_theta, -1.0):
        orthogonal_vector = np.array([1.0, 0.0, 0.0]) if not np.isclose(v1[0], 1.0) else np.array([0.0, 1.0, 0.0])
        orthogonal_vector -= v1 * np.dot(orthogonal_vector, v1)
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)
        return -np.eye(3) + 2 * np.outer(orthogonal_vector, orthogonal_vector)
    cross_prod_matrix = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])
    rotation_matrix = (
        np.eye(3) + cross_prod_matrix + 
        (np.dot(cross_prod_matrix, cross_prod_matrix) * ((1 - cos_theta) / (sin_theta ** 2)))
    )
    return rotation_matrix

def str_to_bool(string):
    if string not in ['True', 'true', '1', 'False', 'false', '0']:
        raise TypeError(f" '{string}' is not convertible to bool")
    return string in ['True', 'true', '1']

def broadcast_to_len(target_len, arg):
    arg_arr = np.asarray(arg, dtype=object)
    if arg_arr.ndim > 0 and len(arg_arr) == target_len: # target lenght
        return arg
    else: # lenghts missmatch
        return [arg] * target_len

class BondWrapper:
    def __init__(self, bond_handle):
        # Store the bond_handle instance
        self._bond_handle = bond_handle

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped object
        return getattr(self._bond_handle, name)

    def __setattr__(self, name, value):
        # Ensure that _bond_handle is set on the wrapper, not on the wrapped object
        if name == "_bond_handle":
            super().__setattr__(name, value)
        else:
            setattr(self._bond_handle, name, value)

    def __delattr__(self, name):
        # Delegate deletion of attributes to the wrapped object
        delattr(self._bond_handle, name)

    def __repr__(self):
        # Customize how the wrapper is printed
        return f"BondWrapper({repr(self._bond_handle)})"
    
    def get_raw_handle(self):
        """
        Returns the raw object being wrapped.
        """
        return self._bond_handle
    
import espressomd

def add_box_constraints_func(sys, wall_type=0, sides=['all'], inter=None, types_=None, object_types=None, bottom=None, top=None, left=None, right=None, back=None, front=None):
    """
    Adds wall constraints to the simulation box along specified sides.

    This method places flat wall constraints (using `espressomd.shapes.Wall`) perpendicular to the box axes, typically used to confine particles within the simulation domain. By default, walls are added on all six faces of the box. You can customize which walls to include or exclude, their positions, and interaction types with other particles.
    By default:
        bottom - z=0; top - z=sys.box_l[2];
        left - y=0  ; right - y=sys.box_l[1];
        back - x=0  ; front - z=sys.box_l[0];

    Parameters
    ----------
    wall_type : int, optional
        Particle type used for the wall (default: 0).
    sides : list of str, optional
        Specifies which sides to add walls on. Default is ['all'], which includes all six box faces.
        Supported values:
            - 'all': add walls on all six faces.
            - 'sides': add walls on all but the top and bottom.
            - Individual sides: 'top', 'bottom', 'left', 'right', 'front', 'back'.
            - 'no-<side>': exclude specific sides, e.g., 'no-top', 'no-right', 'no-sides'.
    inter : str or list of str, optional
        Type(s) of interaction to enable between wall and specified particle types. Currently supports:
            - 'wca': Weeks–Chandler–Andersen potential with large epsilon.
    types_ : list of int, optional
        Particle types that will interact with the walls. If None, all non-wall types in the system are used.
    bottom, top, left, right, back, front : float, optional
        Position of each wall, defined as the distance to the xOy plane (for top/bottom), xOz plane (for left/right),
        or yOz plane (for front/back). If not specified, the position defaults to the corresponding boundary of the simulation box.


    Returns
    -------
    list of espressomd.constraints.ShapeBasedConstraint
        List of wall constraint objects added to the system. (can be used to later specify which walls to remove).
        Organized as: bottom->top->left->right->back->front

    Notes
    -----
    - If `sides` includes any entry starting with 'no-', that side will be excluded even if 'all' or 'sides' is specified.
    - The wall interaction can be configured by specifying `inter` and, optionally, `types_`.
    - Walls are defined using outward-pointing normals and placed at specified distances from the origin.
    - The method adds constraints to `sys.constraints` directly.
    """
    try:
        PartDictSafe({'wall': wall_type})
    except:
        raise ValueError("wall_type must be unique from all other particle types. Default is 0.")

    sides = np.array([sides]).ravel().tolist()
    if "no-" in sides[0]:
        sides.append('all')

    if bottom is None:
        bottom = 0
    else:
        sides.append('bottom')
    if top is None:
        top = sys.box_l[2]
    else:
        sides.append('top')
    if left is None:
        left = 0
    else:
        sides.append('left')
    if right is None:
        right = sys.box_l[1]
    else:
        sides.append('right')
    if back is None:
        back = 0
    else:
        sides.append('back')
    if front is None:
        front = sys.box_l[0]
    else:
        sides.append('front')

    wall_constraints = []

    ###########################
    # top - bottom - const. z #
    ###########################
    if 'bottom' in sides or ('all' in sides and 'no-bottom' not in sides):
        wall = espressomd.shapes.Wall(dist=bottom, normal=[0,0,1])
        wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
        sys.constraints.add(wall_constraint)
        wall_constraints.append(wall_constraint)
    if 'top' in sides or ('all' in sides and 'no-top' not in sides):
        wall = espressomd.shapes.Wall(dist=-top, normal=[0,0,-1])
        wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
        sys.constraints.add(wall_constraint)
        wall_constraints.append(wall_constraint)
    if 'no-sides' not in sides:
        ###########################
        # left - right - const. y #
        ###########################
        if 'left' in sides or ('sides' in sides and 'no-left' not in sides) or ('all' in sides and 'no-left' not in sides):
            wall = espressomd.shapes.Wall(dist=left, normal=[0,1,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)
        if 'right' in sides or ('sides' in sides and 'no-right' not in sides) or ('all' in sides and 'no-right' not in sides):
            wall = espressomd.shapes.Wall(dist=-right, normal=[0,-1,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)
        ###########################
        # back - front - const. x #
        ###########################
        if 'back' in sides or ('sides' in sides and 'no-back' not in sides) or ('all' in sides and 'no-back' not in sides):
            wall = espressomd.shapes.Wall(dist=back, normal=[1,0,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)
        if 'front' in sides or ('sides' in sides and 'no-front' not in sides) or ('all' in sides and 'no-front' not in sides):
            wall = espressomd.shapes.Wall(dist=-front, normal=[-1,0,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)

    # set interactions
    if inter is not None:
        inter= np.array([inter]).ravel()

        if types_ is None:
            if object_types is None:
                types_= set([type_ for type_ in sys.part.all().type if type_ != wall_type])
            else:
                types_ = set([ele.part_types['real'] for ele in object_types])
        else:
            types_= np.array([types_]).ravel()

        if 'wca' in inter:
            for type_ in types_:
                sigma = sys.non_bonded_inter[type_,type_].wca.sigma/2 / 2**(1/6)
                if sigma < 0.001:
                    raise ValueError(f"Interaction of type {type_} with wall is 0, has these particles have no interaction defined. If you would like to have no interactions between particles, but only with wall, then hange this function or do it with normal espresso constraints.")
                sys.non_bonded_inter[wall_type,type_].wca.set_params(epsilon=1E6, sigma=sigma)

    return wall_constraints

def remove_box_constraints_func(sys, wall_type=0, wall_constraints=None, part_types=None, object_types=None):
    """ Removes wall_constraints from system. Default: removes all espressomd.shapes.Wall constraints.
        If part_types is not None, remove only interactions with those particle types.
    system
    list of espressomd.constraints.ShapeBasedConstraint wall_constraints
    list of particles types to stop interactoin with box part_types
    """
    system_constraints = list(sys.constraints)
    if wall_constraints is None:
        wall_constraints = [constraint for constraint in system_constraints
                            if ( isinstance(constraint, espressomd.constraints.ShapeBasedConstraint) and isinstance(constraint.shape, espressomd.shapes.Wall)
                            and ( constraint.particle_type == wall_type or wall_type == 'all') ) ]
    else:
        wall_constraints = np.array([wall_constraints]).ravel()

        
    if part_types is None and object_types is None: #removes actual cosntraints (removes interactions, if no more walls of that type)
        part_types= set([type_ for type_ in sys.part.all().type])

        original_wall_types = set([constraint.particle_type for constraint in system_constraints])
        for wall in wall_constraints: #remove walls
            sys.constraints.remove(wall)
        leftover_wall_types = set([constraint.particle_type for constraint in list(sys.constraints)])
        box_types_remove = original_wall_types - leftover_wall_types
    elif part_types is None: # removes only interactions (based on objects)
        object_types = np.array([object_types]).ravel()
        part_types = set([ele.part_types[typ] for ele in object_types for typ in ele.part_types])
    else: # removes only interactions (based on part_types)
        box_types_remove = set([constraint.particle_type for constraint in wall_constraints])
        part_types = np.array([part_types]).ravel()

    # remove inter for specific types
    for box_type in box_types_remove:
        for type_ in part_types:
            sys.non_bonded_inter[box_type, type_].reset()

def check_free_cuboid(sys, cuboid_l, cuboid_l_shift=None):
    if cuboid_l_shift is None:
        cuboid_l_shift = np.zeros((3))
    pos = sys.part.all().pos
    if len(pos) == 0:
        return True
    else:
        return np.all(np.any((pos < cuboid_l_shift) | (pos > cuboid_l_shift + cuboid_l), axis=1))

def avoid_explosion(sys, F_TOL, MAX_STEPS=5, F_incr=100, I_incr=100):
        """
        Iteratively caps forces to prevent simulation instabilities.
        :param F_TOL: float | Force change tolerance between iterations to determine convergence.
        :param MAX_STEPS: int | Maximum number of steps for force iteration. Default is 5.
        :param F_incr: int | Amount to increase force cap by each iteration. Default is 100.
        :param I_incr: int | Amount to increase integration steps by each iteration. Default is 100.
        :return: None

        The method gradually increases both the force cap and integration timestep while monitoring the relative force change between iterations. If the relative change falls below F_TOL or MAX_STEPS is reached, the iteration stops.
        """
        timestep_og=sys.time_step
        timestep_icr=timestep_og/MAX_STEPS
        logging.info('iterating with a force cap.')
        sys.integrator.run(0)
        STEP=1
        while True:
            sys.time_step=timestep_icr*STEP
            old_force = np.max(np.linalg.norm(
                sys.part.all().f, axis=1))
            sys.force_cap = F_incr
            sys.integrator.run(I_incr)
            force = np.max(np.linalg.norm(sys.part.all().f, axis=1))
            rel_force = np.abs((force - old_force) / old_force)
            logging.info(f'rel. force change: {rel_force:.2e}')
            if (rel_force < F_TOL) or (STEP >= MAX_STEPS):
                break
            STEP += 1
            I_incr += I_incr
            F_incr += F_incr

        sys.force_cap = 0
        sys.time_step=timestep_og
        logging.info('explosions avoided sucessfully!')
import h5py
import numpy as np
import warnings

class H5DataSelector:
    """
    A simplified interface to access simulation data stored in an HDF5 file. The H5DataSelector maintains internal slice information for timesteps (axis 0) and particles (axis 1) and supports chaining via its accessor properties:
    
      - .timestep: Provides an interface for slicing/iterating over timesteps.
      - .particles: Provides an interface for slicing/iterating over particles.

    **Usage Examples:**

      # Slice timesteps 10 to 20 (inclusive of the start, exclusive of the stop)
      time_subset = data.timestep[10:20]

      # Slice particles 0 to 100
      particle_subset = data.particles[:100]

      # Chain slicing:
      sub = data.timestep[:5].particles[0:10]

      # Retrieve property data (e.g., positions):
      pos = data.pos  # This calls get_property("pos")

      # Connectivity-based particle selection:
      filament_particles = data.select_particles_by_object("Filament", connectivity_value=0)
      monomer_subset = filament_particles.particles[10:20].timestep[5]

    **Notes:**
      - Direct indexing on a top-level H5DataSelector is disallowed to ensure explicit axis selection. Use the accessor properties (.timestep or .particles) for slicing.
      - Iteration and len() are only defined on the accessor objects.
    
    **Raises:**
      - ValueError if the dimensions of the stored property datasets are inconsistent.
    """
    def __init__(self, h5_file, particle_group, ts_slice=None, pt_slice=None):
        self.h5_file = h5_file
        self.particle_group = particle_group  # e.g., "Filament"
        self.metadata = self._build_h5_tree(h5_file)
        # Validate that all properties in the given particle group share the same (timesteps, particles)
        self.common_dims = self.sanity_check(self.metadata, self.particle_group)

        # Set default slices if not provided (select all timesteps/particles)
        self.ts_slice = ts_slice if ts_slice is not None else slice(None)
        self.pt_slice = pt_slice if pt_slice is not None else slice(None)

        # Get step tuple (from particle 0)
        self.steps_tuple = h5_file[f"particles/{particle_group}/id/step"][self.ts_slice]
        
        # Set sys group in its little wrapper to be nicer and more seperate from the rest of the suspicious looking groups - like connectivity, for real, what is that supossed to mean. We are indeed all connected, in a way, I guess, so why separate connections based on some set and arbitrary rule. And I stopped there. Long day of coding.
        # TO IMPLEMENT

    def __getitem__(self, key):
        raise TypeError(
            "Direct indexing on a H5DataSelector is not allowed. "
            "Use the 'timestep' or 'particles' accessor for slicing instead."
        )

    def _build_h5_tree(self, obj):
        """
        Recursively builds a nested dictionary representing the HDF5 file structure.

        For groups:
            Returns a dict with keys for each member, and a '_meta' key containing:
              - type: 'Group'
              - members: list of member names

        For datasets:
            Returns a dict containing:
              - type: 'Dataset'
              - shape: dataset.shape
              - dtype: str(dataset.dtype)
              - attributes: dict of dataset attributes (if any)

        Args:
            obj (h5py.Group or h5py.Dataset): HDF5 object to build the structure from.

        Returns:
            dict: A nested dictionary representing the HDF5 structure.
        """
        tree = {}
        if isinstance(obj, h5py.Group):
            tree['_meta'] = {
                'type': 'Group',
                'members': list(obj.keys())
            }
            for key, item in obj.items():
                tree[key] = self._build_h5_tree(item)
        elif isinstance(obj, h5py.Dataset):
            tree = {
                'type': 'Dataset',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'attributes': {attr: obj.attrs[attr] for attr in obj.attrs}
            }
        return tree

    def sanity_check(self, metadata, particle_group):
        """
        Ensures that all properties in a given particle group have consistent timestep and particle dimensions.

        Args:
            metadata (dict): The hierarchical dictionary of the HDF5 structure.
            particle_group (str): The name of the particle group.

        Returns:
            tuple: (timesteps, particles) common to all valid 'value' datasets.

        Raises:
            ValueError: If the particle group is not found or dimensions are inconsistent.
        """
        try:
            particle_meta = metadata["particles"][particle_group]
        except KeyError:
            raise ValueError(f"Particle group '{particle_group}' not found in metadata. \n metadata: {metadata.keys()} \n metadata/particles: {metadata["particles"].keys()}")

        reference_dims = None  # Expected dimensions as (timesteps, particles)
        for prop, prop_info in particle_meta.items():
            if prop == "_meta":
                continue
            if not (isinstance(prop_info, dict) and "value" in prop_info):
                continue

            if prop == "bonds":
                assert prop_info["value"].get("shape")[0] <= 6
                continue

            value_info = prop_info["value"]
            shape = value_info.get("shape")
            if shape is None or len(shape) < 2:
                continue

            current_dims = shape[:2]
            if reference_dims is None:
                reference_dims = current_dims
            else:
                if reference_dims != current_dims:
                    raise ValueError(
                        f"Inconsistent shape for property '{prop}': expected timesteps,particles={reference_dims}, got {current_dims}"
                    )
        if reference_dims is None:
            raise ValueError("No valid 'value' datasets found in particle group for sanity check.")
        return reference_dims

    def __iter__(self):
        raise TypeError("H5DataSelector objects are not iterable. Use the '.timestep' or '.particles' accessor for iteration.")

    def __len__(self):
        raise TypeError("len() is ambiguous on H5DataSelector objects. Use '.timestep' or '.particles' accessor to get the length of the relevant axis.")

    @property
    def timestep(self, step=None):
        """
        Accessor for slicing/iterating over the timestep axis.

        Returns:
            TimestepAccessor: An accessor object for timestep operations.

        Usage:
            # Slicing timesteps 10 to 20:
            data.timestep[10:20]

            # Iterating over each timestep in a slice:
            for t in data.timestep[5:10]:
                process(t)
        """
        return TimestepAccessor(self)
        
    def step(self, step):
        """
        Accessor for slicing/iterating over the timestep axis, with the simulation step values, insted of indexes.

        Args:
            step (int or range of ints): The simulation steps to slice. If None, use dataset indices

        Returns:
            TimestepAccessor: An accessor object for timestep operations.

        Usage:
            # Getting timestep corresponding to step 1E6
            data.timestep(1000000)

            # Slicing timesteps from simulation steps (0,100,500,1000):
            data.timestep((0,100,500,1000))


        """
        matches = np.flatnonzero(np.isin(self.steps_tuple, np.atleast_1d(step)))
        return self.timestep[tuple(matches)]

    @property
    def particles(self):
        """
        Accessor for slicing/iterating over the particle axis.

        Returns:
            ParticleAccessor: An accessor object for particle operations.

        Usage:
            # Slicing particles 0 to 100:
            data.particles[0:100]

            # Iterating over each particle in a slice:
            for p in data.particles[20:30]:
                process(p)
        """
        return ParticleAccessor(self)

    def get_property(self, prop):
        """
        Retrieve data for a given property from the HDF5 dataset applying the current slices.

        Args:
            prop (str): The property name (e.g., 'pos', 'f').

        Returns:
            ndarray: The dataset with the applied timestep and particle slices.
        """
        ds_path = f"particles/{self.particle_group}/{prop}/value"
        ds = self.h5_file[ds_path]
        if prop != "bonds": # most properties have shape (_,_,3 or 1)
            if ds.shape[2] == 1: # properly ouput things like id or type, in their correct type, not in a list
                return ds[self.ts_slice, self.pt_slice, 0]
            else:
                return ds[self.ts_slice, self.pt_slice, :]
        
        else: # for bonds (special case, because of vlen arrays)
            if hasattr(self.ts_slice, '__iter__'):
                ts_slice = self.ts_slice
            else:
                ts_slice = [self.ts_slice]
            if hasattr(self.pt_slice, '__iter__'):
                pt_slice = self.pt_slice
            else:
                pt_slice = [self.pt_slice]

            bond_list_all_ts_slice=[]
            for ts in ts_slice:
                bond_list_single_ts=[]
                for pt in pt_slice:
                    try:
                        assert len(ds[ts, pt]) > 0
                        bond_list_single_ts.append(ds[ts, pt])
                    except:
                        bond_list_single_ts.append([])
                bond_list_all_ts_slice.append(bond_list_single_ts)
            return np.asarray(bond_list_all_ts_slice)
    
    def get_connectivity_values(self, object_name, predicate=None, fast=False):
        """
        Return the raw connectivity pairs for a given object, using the
        ParticleHandle_to_<object_name> dataset.

        Parameters
        ----------
        object_name : str
            Name of the connected object type (e.g. "Filament").

        Returns
        -------
        ndarray of shape (N, 2)
            Array of [particle_id, object_index] pairs.
        """
        ds_path = f"connectivity/{self.particle_group}/ParticleHandle_to_{object_name}"
        obj_ids=self.h5_file[ds_path][:,-1]
        ids=np.unique(obj_ids)
        ret_ids=[]
        def divide_and_conquer():
            nonlocal obj_ids, ids, predicate
            left=0
            right=len(ids)
            while left<right:
                mid = (left+right) // 2
                filter_mask = obj_ids==ids[mid]
                particle_indices = np.flatnonzero(filter_mask)
                particle_indices = particle_indices.tolist()
                subset = H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices)
                if predicate(subset):
                    left=mid+1
                else:
                    right=mid
            return left
        if predicate is not None:
            if fast:
                up_from_me=divide_and_conquer()
                ret_ids=ids[:up_from_me]
            else:
                for i in ids:
                    filter_mask = obj_ids==i
                    particle_indices = np.flatnonzero(filter_mask)
                    particle_indices = particle_indices.tolist()
                    subset = H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices)
                    if predicate(subset):
                        ret_ids.append(i)
                ret_ids=np.array(ret_ids)
                
        else:
            ret_ids=ids
        return ret_ids

    def select_particles_by_object(self, object_name, connectivity_value=None,predicate=None):
        """
        Select a subset of particles based on a connectivity dataset. The indices are sorted and stored as a list (for correct slicing behavior).

        Args:
            object_name (str): Name of the connectivity object (e.g., "Filament").
            connectivity_value ([int, float or None], optional): The value to match in the connectivity map. Defaults to all values: None.

        Returns:
            H5DataSelector: A new selector with the particle slice set to the selected indices.
        """
        ds_name = f"connectivity/{self.particle_group}/ParticleHandle_to_{object_name}"
        connectivity_map = self.h5_file[ds_name][:,1]
        if connectivity_value is None:
            filter_mask = connectivity_map
        else:
            connectivity_value=np.atleast_1d(connectivity_value)
            filter_mask = np.isin(connectivity_map, connectivity_value)
        particle_indices = np.flatnonzero(filter_mask)
        subset=H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices.tolist())
        if predicate is not None:
            mask=predicate(subset).flatten()
            particle_indices=particle_indices[mask]
            subset=H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices.tolist())
        return subset
    
    def select_particles_by_predicate(self, object_name, predicate):
        """
        Select particles linked to a connectivity object by applying a predicate,
        without pre-filtering by a connectivity value.

        Args:
            object_name (str): Name of the connectivity object (e.g., "Filament").
            predicate (callable): Function taking an H5DataSelector and returning a boolean mask.

        Returns:
            H5DataSelector: Selector with particle slice set to the indices passing the predicate.
        """
        ds_name = f"connectivity/{self.particle_group}/ParticleHandle_to_{object_name}"

        # Take ALL particles represented by rows of the connectivity dataset (row index == particle index).
        n_rows = self.h5_file[ds_name].shape[0]
        particle_indices = np.arange(n_rows, dtype=int)

        # Build subset over all those particles, then apply predicate.
        subset = H5DataSelector(self.h5_file, self.particle_group,
                                ts_slice=self.ts_slice, pt_slice=particle_indices.tolist())
        mask = np.asarray(predicate(subset)).flatten()

        # Keep only the particle indices where the predicate is True.
        selected = particle_indices[mask]

        return H5DataSelector(self.h5_file, self.particle_group,
                            ts_slice=self.ts_slice, pt_slice=selected.tolist())


    def get_connectivity_map(self, parent_key, child_key):
        """
        Retrieve the connectivity map between parent and child objects from the HDF5 file.

        This method constructs a dataset path based on the particle group and the specified parent and child keys, then attempts to retrieve an array of [parent_id, child_id] pairs from the HDF5 connectivity group. If the dataset is not found, a warning is issued and None is returned.

        Parameters
        ----------
        parent_key : str
            Name of the parent object type (e.g. "Filament").
        child_key : str
            Name of the child object type (e.g. "Quadriplex").

        Returns
        -------
        ndarray of shape (N, 2) or None
            An array containing [parent_id, child_id] pairs if the connectivity map exists; otherwise, None.
        """
        ds_path = f"connectivity/{self.particle_group}/{parent_key}_to_{child_key}"
        return_map=None
        try:
            return_map = self.h5_file[ds_path][:]
        except KeyError:
            warnings.warn(f"Connectivity map '{ds_path}' not found in HDF5 file.")
        return return_map

    def get_child_ids(self, parent_key, child_key, parent_id):
        """
        Retrieve a sorted list of child object IDs connected to a given parent_id.

        This method first attempts to obtain the connectivity map relating the parent and child objects. If the connectivity map is not found (i.e., None), a warning is issued and None is returned. Otherwise, it filters the connectivity entries to select rows with the matching parent_id,
        extracts the corresponding child IDs, converts them to integers, sorts them, and returns the resulting list.

        Parameters
        ----------
        parent_key : str
            Name of the parent object type.
        child_key : str
            Name of the child object type.
        parent_id : int
            The identifier for the parent object.

        Returns
        -------
        List[int] or None
            A sorted list of child object IDs connected to the given parent_id.
            Returns None if the connectivity map is missing.
        """
        conn = self.get_connectivity_map(parent_key, child_key)
        return_map = None
        if conn is None:
            warnings.warn(f"No children with key {child_key} found for parent with key {parent_key}.")
        else:
            # Filter rows matching the parent_id, then extract child IDs
            child_ids = conn[conn[:, 0] == parent_id, 1]
            return_map = sorted(int(cid) for cid in child_ids)
            # Convert to Python ints and sort
        return return_map

    def get_parent_ids(self, parent_key, child_key, child_id):
        """
        Return a sorted list of parent object IDs connected to a given child_id.

        Parameters
        ----------
        parent_key : str
            Name of the parent object type.
        child_key : str
            Name of the child object type.
        child_id : int
            The who_am_i identifier of the child object.

        Returns
        -------
        List[int]
            Sorted list of parent who_am_i IDs that link to the child.
        """
        conn = self.get_connectivity_map(parent_key, child_key)
        parent_ids = conn[conn[:, 1] == child_id, 0]
        return sorted(int(pid) for pid in parent_ids)

    def __getattr__(self, attr):
        """
        Delegate attribute access to property retrieval.

        This allows a shorthand for accessing data properties such that:
            data.pos
        is equivalent to:
            data.get_property('pos')

        Args:
            attr (str): The attribute name.

        Returns:
            The property data if available.

        Raises:
            AttributeError: If the property does not exist.
        """
        try:
            return self.__dict__[attr]
        except KeyError:
            return self.get_property(attr)

    def __repr__(self):
        return (f"<H5DataSelector(particle_group={self.particle_group}, "
                f"ts_slice={self.ts_slice}, pt_slice={self.pt_slice})>")


class TimestepAccessor:
    """
    An accessor for slicing and iterating over timesteps of a H5DataSelector.

    It composes new timestep slices with any existing slice and enables iteration
    where each iteration yields a H5DataSelector corresponding to a single timestep.
    """
    def __init__(self, sim_data):
        """
        Initialize the TimestepAccessor.

        Args:
            sim_data (H5DataSelector): The parent data selector instance.
        """
        self.sim_data = sim_data

    def __getitem__(self, key):
        """
        Compose the new timestep index with the current slice.

        Args:
            key (int, slice, or list/tuple): The new timestep index or slice.

        Returns:
            H5DataSelector: A new selector with the updated timestep slice.

        Raises:
            IndexError: If the new index is out of bounds relative to the effective timestep indices.
        """
        total_timesteps = self.sim_data.common_dims[0]
        composed = _compose_index(self.sim_data.ts_slice, key, total_timesteps)
        return H5DataSelector(self.sim_data.h5_file, self.sim_data.particle_group, ts_slice=composed, pt_slice=self.sim_data.pt_slice)

    def __iter__(self):
        """
        Iterate over the effective timestep indices.

        Yields:
            H5DataSelector: Each selector has ts_slice set to a single timestep.
        """
        total_timesteps = self.sim_data.common_dims[0]
        ts_slice = self.sim_data.ts_slice
        if isinstance(ts_slice, slice):
            indices = list(range(*ts_slice.indices(total_timesteps)))
        elif isinstance(ts_slice, (list, tuple)):
            indices = ts_slice
        else:
            indices = [ts_slice]
        for idx in indices:
            yield H5DataSelector(self.sim_data.h5_file, self.sim_data.particle_group, ts_slice=idx, pt_slice=self.sim_data.pt_slice)

    def __len__(self):
        """
        Return the number of timesteps in the current slice.

        Returns:
            int: The count of timesteps.
        """
        total_timesteps = self.sim_data.common_dims[0]
        ts_slice = self.sim_data.ts_slice
        if isinstance(ts_slice, slice):
            start, stop, step = ts_slice.indices(total_timesteps)
            return len(range(start, stop, step))
        elif isinstance(ts_slice, (list, tuple)):
            return len(ts_slice)
        elif isinstance(ts_slice, int):
            return 1
        else:
            return total_timesteps

    def __repr__(self):
        return f"<TimestepAccessor(ts_slice={self.sim_data.ts_slice})>"


class ParticleAccessor:
    """
    An accessor for slicing and iterating over particles of a H5DataSelector.

    It composes new particle indices with any existing slice and enables iteration
    where each iteration yields a H5DataSelector corresponding to a single particle index.
    """
    def __init__(self, sim_data):
        """
        Initialize the ParticleAccessor.

        Args:
            sim_data (H5DataSelector): The parent data selector instance.
        """
        self.sim_data = sim_data

    def __getitem__(self, key):
        """
        Compose the new particle index with the current slice.

        Args:
            key (int, slice, or list/tuple): The new particle index or slice.

        Returns:
            H5DataSelector: A new selector with the updated particle slice.

        Raises:
            IndexError: If the new index is out of range for the effective particle indices.
        """
        total_particles = self.sim_data.common_dims[1]
        composed = _compose_index(self.sim_data.pt_slice, key, total_particles)
        return H5DataSelector(self.sim_data.h5_file, self.sim_data.particle_group, ts_slice=self.sim_data.ts_slice, pt_slice=composed)

    def __iter__(self):
        """
        Iterate over the effective particle indices.

        Yields:
            H5DataSelector: Each selector has pt_slice set to a single particle index.
        """
        total_particles = self.sim_data.common_dims[1]
        pt = self.sim_data.pt_slice
        if isinstance(pt, slice):
            indices = list(range(*pt.indices(total_particles)))
        elif isinstance(pt, (list, tuple)):
            indices = pt
        else:
            indices = [pt]
        for idx in indices:
            yield H5DataSelector(self.sim_data.h5_file, self.sim_data.particle_group, ts_slice=self.sim_data.ts_slice, pt_slice=idx)

    def __len__(self):
        """
        Return the number of particles in the current slice.

        Returns:
            int: The count of particles.
        """
        total_particles = self.sim_data.common_dims[1]
        pt_slice = self.sim_data.pt_slice
        if isinstance(pt_slice, slice):
            start, stop, step = pt_slice.indices(total_particles)
            return len(range(start, stop, step))
        elif isinstance(pt_slice, (list, tuple)):
            return len(pt_slice)
        elif isinstance(pt_slice, int):
            return 1
        else:
            return total_particles

    def __repr__(self):
        return f"<ParticleAccessor(pt_slice={self.sim_data.pt_slice})>"


def _compose_index(existing, new, total_length):
    """
    Compose two layers of indexing on a given axis by converting the existing index into an explicit list,
    then applying the new index. This ensures that chained indexing works similarly to Python's native list slicing.

    Args:
        existing (int, slice, or list/tuple): The current index (or composed indices).
        new (int, slice, or list/tuple): The new index to be applied.
        total_length (int): The full length of the axis in the underlying dataset.

    Returns:
        int, list, or tuple: The composed index representing the effective selection on the axis.

    Raises:
        IndexError: If the new index is out of bounds for the effective indices.
        TypeError: If unsupported types are provided for indexing.
    """
    # Convert the existing index to an explicit list.
    if isinstance(existing, slice):
        base = list(range(*existing.indices(total_length)))
    elif isinstance(existing, (list, tuple)):
        base = list(existing)
    elif isinstance(existing, int):
        base = [existing]
    else:
        raise TypeError("Unsupported type for the existing index.")

    # Apply the new indexing on the explicit list.
    if isinstance(new, int):
        try:
            result = base[new]  # Raises IndexError if out-of-bounds.
        except IndexError as e:
            raise IndexError(
                f"Index {new} is out of range for composed indices of length {len(base)}"
            ) from e
    elif isinstance(new, slice):
        result = base[new]
    elif isinstance(new, (list, tuple)):
        result = [base[i] for i in new]
    else:
        raise TypeError("Unsupported type for the new index.")
    return result
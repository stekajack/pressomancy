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

      # Predicate-based particle selection:
      final_type = data.select_particles_by_object(
          "Filament",
          connectivity_value=0,
          predicate=lambda subset: subset.type == 1,
      )

    **Notes:**
      - Direct indexing on a top-level H5DataSelector is disallowed to ensure explicit axis selection. Use the accessor properties (.timestep or .particles) for slicing.
      - `.timestep[...]` slices by frame index on axis 0. It does not query the stored HDF5 `step` or `time` datasets.
      - Iteration and len() are only defined on the accessor objects.
      - Connectivity predicates are evaluated on the current H5DataSelector view. They select particles, not timesteps: the returned selector preserves the current timestep slice and only narrows the particle slice.
    
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
              - attributes: dict of group attributes

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
                'members': list(obj.keys()),
                'attributes': {attr: obj.attrs[attr] for attr in obj.attrs}
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
            raise ValueError(f"Particle group '{particle_group}' not found in metadata.")

        reference_dims = None  # Expected dimensions as (timesteps, particles)
        for prop, prop_info in particle_meta.items():
            if prop == "_meta":
                continue
            if not (isinstance(prop_info, dict) and "value" in prop_info):
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
    def timestep(self):
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
        time_selected = ds[self.ts_slice, :, :]
        return time_selected[:, self.pt_slice, :] if time_selected.ndim == 3 else time_selected[self.pt_slice, :]
    
    def get_box(self):
        """Return fixed-box metadata for the current particle group."""
        box_path = f"particles/{self.particle_group}/box"
        box_group = self.h5_file[box_path]
        dimension = int(box_group.attrs["dimension"])
        boundary_raw = np.atleast_1d(box_group.attrs["boundary"]).tolist()
        boundary = tuple(
            item.decode("ascii") if isinstance(item, bytes) else str(item)
            for item in boundary_raw
        )
        edges = np.asarray(box_group["edges"][:], dtype=np.float64)
        return {
            "dimension": dimension,
            "boundary": boundary,
            "edges": edges,
        }

    def get_connectivity_values(self, object_name, predicate=None, fast=False):
        """
        Return object IDs present in ``ParticleHandle_to_<object_name>``.

        When a predicate is provided, each candidate object ID is first mapped
        to the particle indices of the current particle group. This keeps the
        returned per-object subset aligned with ``particles/<particle_group>``
        even when ``ParticleHandle_to_<object_name>`` is not in that particle
        group's storage order.

        Connectivity IDs are static. The predicate is therefore an object-ID
        filter: it may inspect particle properties on the per-object subset, but
        it must return one scalar truth value for the candidate object ID. If it
        inspects time-dependent arrays, reduce them explicitly with ``any``,
        ``all``, or an explicit timestep selection.

        Parameters
        ----------
        object_name : str
            Name of the connected object type (e.g. "Filament").
        predicate : callable, optional
            Function that accepts a per-object H5DataSelector subset and returns
            True when that object ID should be kept. The subset is provided only
            for inspection and preserves the current selector's timestep slice.
        fast : bool, optional
            If True, assume the predicate is prefix-monotone over the sorted
            object IDs and use a divide-and-conquer search for the cutoff.

        Returns
        -------
        ndarray of shape (N,)
            Array of object IDs.
        """
        ds_path = f"connectivity/{self.particle_group}/ParticleHandle_to_{object_name}"
        ds_father_path = f"connectivity/{self.particle_group}/ParticleHandle_to_{self.particle_group}"
        obj_connectivity = self.h5_file[ds_path][:]
        obj_ids = obj_connectivity[:,-1]
        father_particle_handles = self.h5_file[ds_father_path][:,0]
        ids=np.unique(obj_ids)
        ret_ids=[]

        def subset_for_object_id(object_id):
            object_particle_handles = obj_connectivity[obj_ids == object_id, 0]
            filter_mask = np.isin(father_particle_handles, object_particle_handles)
            particle_indices = np.flatnonzero(filter_mask).tolist()
            return H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices)

        def divide_and_conquer():
            nonlocal obj_ids, ids, predicate
            left=0
            right=len(ids)
            while left<right:
                mid = (left+right) // 2
                subset = subset_for_object_id(ids[mid])
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
                    subset = subset_for_object_id(i)
                    if predicate(subset):
                        ret_ids.append(i)
                ret_ids=np.array(ret_ids)
                
        else:
            ret_ids=ids
        return ret_ids
    
    def select_particles_by_object(self, object_name, connectivity_value, predicate=None):
        """
        Select particles belonging to one or more connected object IDs.

        The connectivity table stores particle handles, so this method maps
        those handles back to indices in ``particles/<particle_group>`` before
        composing a new particle slice. The current timestep slice is preserved.

        A predicate can further narrow the selected particles. It is evaluated on
        the current H5DataSelector subset, not on individual particle objects.
        Predicate masks select particles only:

        - a 1D mask must have shape ``(n_particles,)``;
        - a 2D mask must have shape ``(n_timesteps, n_particles)`` and is
          reduced with ``all`` over the current timestep context.

        This means ``predicate=lambda subset: subset.type == value`` keeps
        particles matching the value at every timestep in the current view,
        while ``predicate=lambda subset: subset.timestep[-1].type == value``
        classifies particles by the last timestep of the current view and then
        returns those particles across the original timestep slice.

        Args:
            object_name (str): Name of the connectivity object (e.g., "Filament").
            connectivity_value (int or float or array-like): Object ID value or
                values to match in the connectivity map.
            predicate (callable, optional): Function taking an H5DataSelector and
                returning a particle mask as described above.

        Returns:
            H5DataSelector: A new selector with the same timestep slice and the
            particle slice set to the selected particle indices.
        """
        # Get particles' ids from connectivity of object_name
        ds_name = f"connectivity/{self.particle_group}/ParticleHandle_to_{object_name}"
        # Get the correct indices from the connectivity of the 'father' object. This is where the particle slices are applied to
        ds_father_name = f"connectivity/{self.particle_group}/ParticleHandle_to_{self.particle_group}"
        connectivity_map = self.h5_file[ds_name][:,1]
        connectivity_value=np.atleast_1d(connectivity_value)
        filter_mask = np.isin(connectivity_map, connectivity_value)
        object_particle_indices = np.ravel(self.h5_file[ds_name][:,0][filter_mask])
        # Get the correct indices for the selected particles from the parent's particle list
        father_particles_indices = self.h5_file[ds_father_name][:,0]
        filter_mask = np.isin(father_particles_indices, object_particle_indices)
        particle_indices = np.flatnonzero(filter_mask)
        subset=H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices.tolist())

        if predicate is not None:
            mask = np.asarray(predicate(subset))
            while mask.ndim > 1 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)
            if mask.ndim == 2:
                mask = np.all(mask, axis=0)
            elif mask.ndim != 1:
                raise ValueError("Predicate must resolve to a particle mask.")
            particle_indices=particle_indices[mask]
            subset=H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices.tolist())
        return subset

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


class H5ObservableSelector:
    """
    A simplified interface to access observable data stored in an HDF5 file.

    The selector maintains internal slice information for the timestep axis only
    and exposes direct access to the stored ``step``, ``time``, and ``value``
    datasets under ``/observables/<name>``.
    """
    def __init__(self, h5_file, observable_name, ts_slice=None):
        self.h5_file = h5_file
        self.observable_name = observable_name
        self.metadata = self._build_h5_tree(h5_file)
        self.common_dims = self.sanity_check(self.metadata, self.observable_name)
        self.ts_slice = ts_slice if ts_slice is not None else slice(None)

    def __getitem__(self, key):
        raise TypeError(
            "Direct indexing on a H5ObservableSelector is not allowed. "
            "Use the 'timestep' accessor for slicing instead."
        )

    def _build_h5_tree(self, obj):
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

    def sanity_check(self, metadata, observable_name):
        try:
            observable_meta = metadata['observables'][observable_name]
        except KeyError:
            raise ValueError(f"Observable '{observable_name}' not found in metadata.")

        try:
            step_shape = observable_meta['step']['shape']
            time_shape = observable_meta['time']['shape']
            value_shape = observable_meta['value']['shape']
        except KeyError as exc:
            raise ValueError(
                f"Observable '{observable_name}' must contain step, time, and value datasets."
            ) from exc

        common_dims = (step_shape[0], time_shape[0], value_shape[0])
        if len(set(common_dims)) != 1:
            raise ValueError(
                f"Observable '{observable_name}' has inconsistent step/time/value lengths: {common_dims}"
            )
        return (value_shape[0],)

    def __iter__(self):
        raise TypeError("H5ObservableSelector objects are not iterable. Use the '.timestep' accessor for iteration.")

    def __len__(self):
        raise TypeError("len() is ambiguous on H5ObservableSelector objects. Use '.timestep' to get the number of selected frames.")

    @property
    def timestep(self):
        return ObservableTimestepAccessor(self)

    @property
    def value(self):
        ds = self.h5_file[f"observables/{self.observable_name}/value"]
        return ds[self.ts_slice, ...]

    @property
    def step(self):
        ds = self.h5_file[f"observables/{self.observable_name}/step"]
        return ds[self.ts_slice]

    @property
    def time(self):
        ds = self.h5_file[f"observables/{self.observable_name}/time"]
        return ds[self.ts_slice]

    def __repr__(self):
        return (f"<H5ObservableSelector(observable_name={self.observable_name}, "
                f"ts_slice={self.ts_slice})>")


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


class ObservableTimestepAccessor:
    """
    An accessor for slicing and iterating over timesteps of a H5ObservableSelector.
    """
    def __init__(self, sim_data):
        self.sim_data = sim_data

    def __getitem__(self, key):
        total_timesteps = self.sim_data.common_dims[0]
        composed = _compose_index(self.sim_data.ts_slice, key, total_timesteps)
        return H5ObservableSelector(self.sim_data.h5_file, self.sim_data.observable_name, ts_slice=composed)

    def __iter__(self):
        total_timesteps = self.sim_data.common_dims[0]
        ts_slice = self.sim_data.ts_slice
        if isinstance(ts_slice, slice):
            indices = list(range(*ts_slice.indices(total_timesteps)))
        elif isinstance(ts_slice, (list, tuple)):
            indices = ts_slice
        else:
            indices = [ts_slice]
        for idx in indices:
            yield H5ObservableSelector(self.sim_data.h5_file, self.sim_data.observable_name, ts_slice=idx)

    def __len__(self):
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
        return f"<ObservableTimestepAccessor(ts_slice={self.sim_data.ts_slice})>"


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

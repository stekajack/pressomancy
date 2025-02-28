class H5DataSelector:
    """
    A simplified interface that wraps an HDF5 file with simulation data.
    Particle properties are stored under:
       particles/{particle_group}/{prop}/value
    where each dataset has shape (timesteps, particles, dim).
    
    This class holds slice information for timesteps (axis 0) and particles (axis 1)
    and supports chaining via its accessor properties.
    """
    def __init__(self, h5_file, particle_group, ts_slice=None, pt_slice=None):
        self.h5_file = h5_file
        self.particle_group = particle_group  # e.g., "Filament"
        # Default slices: all timesteps and all particles.
        self.ts_slice = ts_slice if ts_slice is not None else slice(None)
        self.pt_slice = pt_slice if pt_slice is not None else slice(None)

    def __getitem__(self, key):
        """
        Allow slicing using standard indexing syntax.
        If key is a tuple, we assume (ts_slice, pt_slice).
        Otherwise, the key applies to the timestep axis.
        """
        if isinstance(key, tuple):
            ts_key, pt_key = key
        else:
            ts_key, pt_key = key, self.pt_slice
        return H5DataSelector(self.h5_file, self.particle_group, ts_slice=ts_key, pt_slice=pt_key)

    @property
    def timestep(self):
        """
        Accessor to slice the timestep axis.
        Usage: data.timestep[-5:] returns a new H5DataSelector with ts_slice updated.
        """
        return TimestepAccessor(self)

    @property
    def particles(self):
        """
        Accessor to slice the particle axis.
        Usage: data.particles[0:100] returns a new H5DataSelector with pt_slice updated.
        """
        return ParticleAccessor(self)

    def get_property(self, prop):
        """
        Retrieve data for a property (e.g., 'pos', 'f', etc.) from the HDF5 dataset,
        applying the current timestep and particle slices.
        """
        ds_path = f"particles/{self.particle_group}/{prop}/value"
        ds = self.h5_file[ds_path]
        return ds[self.ts_slice, self.pt_slice, :]

    def select_particles_by_object(self, object_name, connectivity_value=0):
        """
        Convenience method to select particles based on a connectivity dataset.
        For example, if you have a dataset at:
            connectivity/ParticleHandle_to_{object_name}
        this method selects the indices where the second column equals connectivity_value.
        """
        ds_name = f"connectivity/ParticleHandle_to_{object_name}"
        connectivity_map = self.h5_file[ds_name][:]
        particle_indices = connectivity_map[connectivity_map[:, 1] == connectivity_value][:, 0]
        particle_indices.sort()
        return H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices)
    
    def __getattr__(self, attr):
        """
        Called when an attribute lookup fails. Here we assume that if the attribute
        is not found on the H5DataSelector instance, it is meant to be a property name.
        For example, accessing `data.pos` is equivalent to `data.get_property('pos')`.
        """
        try:
            # Avoid interfering with special attributes.
            return self.__dict__[attr]
        except KeyError:
            # Try to return the property data.
            return self.get_property(attr)

    def __repr__(self):
        return (f"<H5DataSelector(particle_group={self.particle_group}, "
                f"ts_slice={self.ts_slice}, pt_slice={self.pt_slice})>")

class TimestepAccessor:
    """
    A simple accessor that operates on a H5DataSelector object to update the timestep slice.
    """
    def __init__(self, sim_data):
        self.sim_data = sim_data

    def __getitem__(self, key):
        # Return a new H5DataSelector with the timestep slice updated.
        return H5DataSelector(self.sim_data.h5_file,
                              self.sim_data.particle_group,
                              ts_slice=key,
                              pt_slice=self.sim_data.pt_slice)

    def __repr__(self):
        return f"<TimestepAccessor(ts_slice={self.sim_data.ts_slice})>"

class ParticleAccessor:
    """
    A simple accessor that operates on a H5DataSelector object to update the particle slice.
    """
    def __init__(self, sim_data):
        self.sim_data = sim_data

    def __getitem__(self, key):
        # Return a new H5DataSelector with the particle slice updated.
        return H5DataSelector(self.sim_data.h5_file,
                              self.sim_data.particle_group,
                              ts_slice=self.sim_data.ts_slice,
                              pt_slice=key)

    def __repr__(self):
        return f"<ParticleAccessor(pt_slice={self.sim_data.pt_slice})>"
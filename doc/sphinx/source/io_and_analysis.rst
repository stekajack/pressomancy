IO And Analysis
===============

Pressomancy's IO layer is useful for the same reason its object model is
useful: it preserves structure. Instead of writing only a flat list of
particle properties, the HDF5 writer stores those properties together with the
ownership and connectivity information that explains which simulation object
each particle came from. That makes the saved file useful in three different
roles at once: as a trajectory, as an analysis source, and as the starting
point for a new simulation state.

Why Use The IO Stack
--------------------

For each registered object family, pressomancy writes particle data under
``/particles/<Group>/<property>`` using the H5MD-style triplet of ``value``,
``step``, and ``time`` datasets. The corresponding property tensors have shape
``(T, N, D)``, where ``T`` is the number of stored frames, ``N`` is the number
of particles in the registered flat view for that family, and ``D`` is the
dimensionality of the property. In parallel, the writer stores object-context
information under ``/connectivity`` so that the file can answer questions a
plain trajectory cannot, such as which particles belong to a given
:class:`~pressomancy.object_classes.filament_class.Filament` or which child
objects belong to a given parent object.

That combined layout is the real benefit of using the built-in IO stack. The
particle-property part is close enough to H5MD to be familiar and efficient,
while the connectivity extension makes the result meaningful inside a
pressomancy workflow. The payoff is practical rather than theoretical: one
format supports long-run storage, object-aware analysis, and source-driven
reconstruction without format conversion in between.

Write And Restart A Trajectory
------------------------------

The usual write workflow starts by telling the simulation which object
families should be persisted. That happens through
:meth:`~pressomancy.simulation.Simulation.inscribe_part_group_to_h5`, which
creates the HDF5 layout and records the flat particle views that later writes
will use. Concretely, the simulation stores those particle handles in
``sim.io_dict['flat_part_view']``. That view is the ordered particle list used
by :meth:`~pressomancy.simulation.Simulation.write_part_group_to_h5` when each
new frame is appended. In other words, ``flat_part_view`` is the bridge
between the simulation-object hierarchy and the flat per-group particle layout
written to HDF5.

.. code-block:: python

   step_index = sim.inscribe_part_group_to_h5(
       group_type=[Filament, Crowder],
       h5_data_path="run.h5",
       mode="NEW",
   )

   for _ in range(n_steps):
       sim.sys.integrator.run(1)
       sim.write_part_group_to_h5(time_step=step_index)
       step_index += 1

On the current branch,
:meth:`~pressomancy.simulation.Simulation.inscribe_part_group_to_h5` supports
four modes. In practice, they are not interchangeable and it is worth being
explicit about what each one assumes.

- ``NEW`` creates a fresh HDF5 file and registers the current flat particle
  view for later writes.
- ``LOAD_NEW`` reopens an existing HDF5 file and rebuilds the flat particle
  view from the connectivity already stored in that file. In practice this is
  the preferred continuation mode when you have both the ESPResSo binary
  checkpoint and the HDF5 file available, because it does not require you to
  reconstruct the full Python-side object graph before continuing IO.
- ``LOAD`` reopens an existing HDF5 file, but it assumes that the simulation
  setup has already been reconstructed in memory and that the current objects
  already own the right particles. This mode therefore depends only on the
  ESPResSo checkpoint for particle state, but it requires the whole simulation
  script to rebuild the object hierarchy before the IO layer can safely resume.
- ``INIT_SRC`` prepares the IO layer for source-driven initialization. It is
  not a continuation mode in the checkpoint sense. It is the mode used when a
  previous HDF5 file should act as the source of positions, orientations, and
  selected properties for a newly constructed simulation.

In day-to-day restart work, ``LOAD_NEW`` is usually the strongest option.

New files also carry optional metadata that complements the particle and
connectivity layout. The writer records H5MD-style root metadata under
``/h5md`` together with pressomancy-specific metadata under
``/parameters/pressomancy``. In practice, that metadata serves two roles.
First, it stores lightweight provenance for the submission script and the
pressomancy checkout that produced the file. Second, it stores the current
``part_types`` map so that ``LOAD_NEW`` can restore symbolic particle-type
bookkeeping directly when that information is available. The
connectivity tables in the HDF5 file give pressomancy enough information to
rebuild the flat particle view directly from saved object ownership, which is
why it offers extra functionality compared with ``LOAD``.

This is also a good place to keep the role of ESPResSo checkpointing in
perspective. The binary checkpoint mechanism is powerful and robust, but it
relies on binary state and pickle-based reconstruction. In practice, that
means it is less portable across systems, package versions, ESPResSo versions,
and runtime layouts such as MPI rank counts. The HDF5 files are less tied to
that execution environment and can, in principle, be used to reconstruct a
simulation state while sidestepping binary checkpoint portability limits. That
does not remove the need for care, but it is one of the reasons the HDF5 path
is so valuable in longer-lived workflows.

Keep Checkpoints And HDF5 In Sync
---------------------------------

The ``force_resize_to_size`` argument exists for a workflow that matters in
practice on HPC platforms. Long simulations often save two independent kinds
of state: an ESPResSo checkpoint for the live system, and an HDF5 trajectory
for analysis. If a job is interrupted between those two writes, the last valid
binary checkpoint and the last valid HDF5 frame may no longer correspond to
one another.

That is why it is a good idea to maintain a global step counter and treat it
as the record of the last reliable state. When that counter says the last safe
frame is smaller than what the HDF5 file currently contains,
``force_resize_to_size`` lets ``LOAD_NEW`` truncate every stored property to
that known-good length before the next write. In other words, it drops ragged
trailing data so that the HDF5 trajectory and the binary checkpoint are again
synchronized.

This is especially useful when a run writes checkpoints and HDF5 frames on
different cadences, or when a failure occurs after one output path succeeded
and the other did not. In that situation, ``LOAD_NEW`` plus
``force_resize_to_size`` gives you a clean way to continue without carrying a
silently inconsistent tail in the HDF5 data.

Analysis API
------------

Reading follows the same explicit style. The entry point is
:class:`~pressomancy.analysis.data_analysis.H5DataSelector`. The important
idea is that the selector never asks you to guess which axis an index belongs
to. Timesteps and particles are separate iteration contexts, exposed through
``.timestep`` and ``.particles``.

.. code-block:: python

   with h5py.File("run.h5", "r") as h5_file:
       data = H5DataSelector(h5_file, particle_group="Filament")

       frame = data.timestep[-1]
       particle_window = data.particles[100:150]
       composed_a = data.timestep[-1].particles[100:150]
       composed_b = data.particles[100:150].timestep[-1]

Because the selector stores one timestep slice and one particle slice
internally, timestep and particle selection can be composed in either order.
That gives you unambiguous iteration contexts for both axes and makes slicing
behavior robust even in longer chained expressions. If you want to iterate
frame by frame, iterate over ``data.timestep``. If you want to iterate over a
particle subset, iterate over ``data.particles``. If you want both, compose
the two contexts first and then access the properties.

Predicates are the next important layer. They let you refine a selected view
without leaving the selector API.

.. code-block:: python

   with h5py.File("run.h5", "r") as h5_file:
       data = H5DataSelector(h5_file, particle_group="Filament")
       frame = data.timestep[-1]

       filament_zero = frame.select_particles_by_object(
           object_name="Filament",
           connectivity_value=0,
           predicate=lambda subset: subset.type == sim.part_types["real"],
       )

       magnetic_filaments = frame.get_connectivity_values(
           "Filament",
           predicate=lambda subset: np.any(subset.dip[..., 2] > 0.0),
       )

This is useful because it lets you express selection logic in the same
coordinate system in which the data are stored. You can first reduce the view
by timestep, then by object ownership, then by a property predicate, without
having to manually reconstruct index arrays outside the API.

The object-context helpers are what make the selector especially useful for
pressomancy-generated data. The selector also exposes the structural metadata
written to the HDF5 file through ``data.metadata``. Group attributes are kept
under ``_meta["attributes"]``, which makes file-level metadata such as
``/h5md`` creator information or ``/parameters/pressomancy/part_types``
accessible without introducing a second analysis API. ``select_particles_by_object`` gives you the
particle subset belonging to one saved object instance, such as one
:class:`~pressomancy.object_classes.filament_class.Filament` or one
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex`.
:meth:`~pressomancy.analysis.data_analysis.H5DataSelector.get_connectivity_values`
lets you enumerate object IDs, optionally filtered by a predicate. Note that it returns object IDs rather than raw connectivity pairs. When ``fast=True``, the predicate is assumed to be prefix-monotone over the sorted object IDs, so the selector can stop at the first failing cutoff instead of scanning every object (binary search).
:meth:`~pressomancy.analysis.data_analysis.H5DataSelector.get_child_ids`,
:meth:`~pressomancy.analysis.data_analysis.H5DataSelector.get_parent_ids`, and
:meth:`~pressomancy.analysis.data_analysis.H5DataSelector.get_connectivity_map`
then let you move up and down the saved object graph. In practice, this enables
workflows such as selecting all particles of one
:class:`~pressomancy.object_classes.filament_class.Filament` at one frame,
querying which
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` objects belong
to that filament, or filtering object IDs based on per-object property tests
without re-deriving connectivity from raw coordinates.

Use Output As A Source
----------------------

Source-driven initialization is best understood as a workflow for building a
new simulation from the structural state of an older one. A representative
example is a long run of fixed-point dipole chains used to reach an
equilibrium self-assembly picture. A later run can then take a chosen snapshot
from that older trajectory, reuse its positions and orientations, and
substitute a different local particle model such as
:class:`~pressomancy.object_classes.egg_model_part.EGGPart` while keeping the
larger assembled structure.

The first step is to declare the source mapping with
:meth:`~pressomancy.simulation.Simulation.set_init_src`.

.. code-block:: python

   sim.set_init_src(
       path="run.h5",
       pos_ori_src_type=["real"],
       type_to_type_map=[("real", "real"), ("virt", "virt")],
       prop_to_prop_map=[("dip", "director")],
   )

In this workflow, positions and orientations are special. When you place
objects through :meth:`~pressomancy.simulation.Simulation.set_objects` with
``mode='INIT_SRC'``, the normal placement step is replaced by source-backed
placement,
and the top-level objects, for example
:class:`~pressomancy.object_classes.filament_class.Filament` instances, are
created directly at the positions and orientations read from the HDF5 file.

.. code-block:: python

   sim.store_objects(objects)
   sim.set_objects(objects, mode="INIT_SRC")

Additional property transfer happens separately through
:meth:`~pressomancy.simulation.Simulation.set_prop_from_src`. This call should
come after the objects have been created and after any object-local setup that
must already exist for the property mapping to make sense, but before the
production dynamics begins. In a chain-suspension workflow, that means the
object hierarchy is built first, anchors and related local structure are
added, and only then are source properties copied onto the newly created
particles.

.. code-block:: python

   sim.store_objects(objects)
   sim.set_objects(objects, mode="INIT_SRC")

   for filament in objects:
       filament.add_anchors(type_name="real")
       filament.bond_anchors()

   sim.set_prop_from_src(objects)

That separation is important. ``set_objects(..., mode='INIT_SRC')`` handles
where the new objects should be placed. ``set_prop_from_src`` handles which
non-positional properties should then be copied from the source snapshot onto
those already created local particles.

A related helper,
:meth:`~pressomancy.simulation.Simulation.mk_src_file`, is useful when the
source you want is hidden inside a larger trajectory. It copies an existing
HDF5 file, collapses it to a single timestep, and can optionally append any
missing per-particle properties needed by the next run. In practice, that is a
clean way to turn a trajectory frame into a reusable source file.

Performance And Common Mistakes
-------------------------------

The HDF5 writer stores each ``value`` dataset with timestep-shaped chunks,
namely ``(1, N, D)``, and uses moderate gzip compression. This is not just a
storage detail. It means a read such as ``data.timestep[-1]`` naturally
touches one frame at a time rather than encouraging whole-trajectory reads.
For long runs, that keeps memory use under control and makes frame-local,
object-aware analysis practical in an interactive workflow.

The most common IO failures are mapping failures rather than storage failures.
If a source-driven workflow breaks, the first things to check are whether
``set_init_src`` was called and whether the declared type names exist both in
the local simulation and in the source file. The next thing to check is the
correspondence between ``type_to_type_map`` and ``prop_to_prop_map``. If those
two lists are not aligned conceptually, the restart logic has no sensible way
to match source data to local particles.

The other recurring mistake is to postpone the structured IO path until the
analysis stage. By that point, the important ownership information may simply
never have been written. If you expect object-level analysis or source-driven
restarts later, the built-in HDF5 path is worth using from the first run.

For a case where hierarchy, placement, and IO all matter at once, continue to
:doc:`g_quadruplex`.

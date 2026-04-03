First Simulation
================

The quickest way to understand pressomancy is to run through one complete
cycle. Create a :class:`~pressomancy.simulation.Simulation`, instantiate one
object family, store it, place it in the box, run a little dynamics, write
one HDF5 frame, and inspect that frame with
:class:`~pressomancy.analysis.data_analysis.H5DataSelector`. This chapter
keeps the scientific model deliberately simple so that the workflow itself is
easy to see. The same sequence later scales to more structured top-level
objects.

Before You Run
--------------

Pressomancy is designed to run inside an ESPResSo-enabled Python environment.
In practice that usually means ``pypresso`` or another interpreter that ships
with ESPResSo and the required compiled features. The minimal example below
uses :class:`~pressomancy.object_classes.part_class.GenericPart`, so it does
not rely on virtual sites or other advanced object features. Many of the more
structured object families do, however, so a properly featured ESPResSo build
is still the normal working assumption for the package as a whole.

Minimal Example
---------------

.. code-block:: python

   import os
   import tempfile
   import numpy as np
   from pressomancy.simulation import Simulation
   from pressomancy.object_classes.part_class import GenericPart
   from pressomancy.analysis import H5DataSelector

   box_dim = np.array([40.0, 40.0, 40.0])

   sim = Simulation(box_dim=box_dim)
   sim.set_sys(timestep=0.01, min_global_cut=3.0)

   part_cfg = GenericPart.config.specify(
       espresso_handle=sim.sys,
       size=1.0,
       espresso_part_kwargs={"type": GenericPart.part_types["real"]},
   )
   particles = [GenericPart(config=part_cfg) for _ in range(100)]

   sim.store_objects(particles)
   sim.set_objects(particles)

   sim.set_steric(key=("real",), wca_eps=1.0, sigma=1.0)
   sim.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim.seed)

   with tempfile.TemporaryDirectory() as tmpdir:
       h5_path = os.path.join(tmpdir, "run.h5")
       step_index = sim.inscribe_part_group_to_h5(
           group_type=[GenericPart],
           h5_data_path=h5_path,
           mode="NEW",
       )

       for _ in range(10):
           sim.sys.integrator.run(10)
           sim.write_part_group_to_h5(time_step=step_index)
           step_index += 1

       data = H5DataSelector(sim.io_dict["h5_file"], particle_group="GenericPart")
       frame = data.timestep[-1]
       positions = frame.pos
       particle_ids = frame.id.flatten()

This example uses the primitive
:class:`~pressomancy.object_classes.part_class.GenericPart` directly rather
than an alias such as
:class:`~pressomancy.object_classes.crowder_class.Crowder`. That keeps the
example as close as possible to the minimal simulation-object contract. The
control flow is still the same one used later for more structured top-level
objects such as filaments, quadriplexes, and telomeric assemblies.

What The Simulation Does
------------------------

The important transition is the one from Python-side object definitions to
actual particles in the ESPResSo system. Calling
:meth:`~pressomancy.simulation.Simulation.store_objects` does not place
anything yet. It registers the objects with the simulation, checks required
features, and updates the shared particle-type bookkeeping that the object and
the simulation manager need to agree on.

Particle creation begins when you call
:meth:`~pressomancy.simulation.Simulation.set_objects`. The simulation first
uses :func:`~pressomancy.helper_functions.partition_cuboid_volume` to generate
candidate regions in the box. That helper uses an FCC lattice as a dense and
regular scaffold for placement. For each chosen region, the simulation calls
the object's ``build_function`` to obtain the local pattern appropriate for
that object. The resulting local coordinates and orientations are then passed
into :meth:`~pressomancy.object_classes.object_class.Simulation_Object.set_object`,
which is where the object finally materializes its particles and any local
connectivity.

With :class:`~pressomancy.object_classes.part_class.GenericPart`, this
stays almost trivial: one object becomes one particle. In a hierarchical
object, the same logic keeps descending. The simulation places only the
top-level object. That object then places its children, and those children
place theirs. The distinction matters because it lets global space management
stay separate from object-specific construction logic.

The write step follows the same philosophy.
:meth:`~pressomancy.simulation.Simulation.inscribe_part_group_to_h5` should be
called before the production loop starts, because that call creates the HDF5
layout and returns the initial global counter used for later writes. Inside
the simulation loop,
:meth:`~pressomancy.simulation.Simulation.write_part_group_to_h5` is then called
periodically with the current counter value, after which the counter is
incremented. Even in a minimal example, that structure is worth keeping,
because it matches the restart-safe pattern used in longer cluster runs.

Choices That Matter
-------------------

The first choice that matters in practice is object ``size``. In pressomancy,
``size`` is part of the contract between the object and the simulation. The
simulation uses it when deciding how much room to allocate for a top-level
object. If the object reports too small a footprint, placement may appear to
succeed while silently setting up avoidable overlaps.

The second choice is whether you are placing a single family of top-level
objects or several. On the current branch,
:meth:`~pressomancy.simulation.Simulation.set_objects` handles one initial
partition cleanly and can then place one additional family relative to the
first. That is sufficient for the shipped examples, but it means top-level
placement should be treated as a deliberate modeling step, not as something
that happens accidentally at the end of object construction.

The third choice is to adopt the HDF5 writer early and structure the loop
around the returned global counter from the start. It is tempting to postpone
structured output until analysis becomes necessary, but that usually means the
useful object context was never written in the first place. If you expect to
ask object-level questions later, continue a run safely, or seed a future run
from the current one, the built-in IO path is already the right one in the
first script.

Common Mistakes
---------------

The two most common beginner mistakes are straightforward. The first is to
forget that ``store_objects`` and ``set_objects`` do different jobs. Storing
registers objects; setting places them and creates particles. If you stop
after the first call, nothing exists in the box yet. The second is to run the
script outside an ESPResSo-capable environment, which tends to surface as
feature or runtime failures rather than as a gentle import error.

Once the objects become more structured, placement failures usually trace back
to incorrect configuration assumptions: ``n_parts`` not matching the number of
child objects, ``size`` not matching the actual occupied footprint, or a local
``build_function`` that is inconsistent with the intended topology. In
source-driven restart workflows, failures are more often caused by mismatched
type or property mappings than by the HDF5 file itself.

For the architectural reasoning behind this workflow, continue to
:doc:`simulation_and_objects`. For the IO side in more detail, continue to
:doc:`io_and_analysis`.

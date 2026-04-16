G-quadruplex Assembly
=====================

This tutorial is a worked example that explains why pressomancy builds from top-level objects, propagates construction downward, and keeps ownership and connectivity in the IO output rather than only particle coordinates. The core build path is
:class:`~pressomancy.object_classes.quadriplex_class.Quartet` ->
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` ->
:class:`~pressomancy.object_classes.filament_class.Filament` or
:class:`~pressomancy.object_classes.tel_sequence.TelSeq`.

The simulation never places quartets directly into the global box. It places a
top-level object and lets that object recursively build the lower levels
beneath it. The two sample workflows on ``main`` make that design concrete: a
G4 Multimer and a folded telomeric sequence.

What Gets Built
---------------

In ``samples/poly_BRACO.py``, :class:`~pressomancy.object_classes.quadriplex_class.Quartet` are the rigid building blocks. Three
quartets are grouped into one
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex`, and several
quadriplexes are grouped into one
:class:`~pressomancy.object_classes.filament_class.Filament`. The filament is
therefore the object the simulation actually places in the box. Once
:meth:`~pressomancy.simulation.Simulation.set_objects` assigns positions and
orientations to the filaments, each filament places its quadriplex children,
and each quadriplex places and bonds its quartet children.

In ``samples/folded_tel_sequence.py``, the lower part of the hierarchy stays
recognizable, but the top-level container changes.
:class:`~pressomancy.object_classes.tel_sequence.TelSeq` creates a telomeric
sequence with fold-specific post-placement logic. Quartets and quadriplexes
are still built in the same general top-down style, but the final local
couplings depend on the chosen telomeric fold type.

G4 Multimers
-------------

The workflow begins by creating a configuration for a :class:`~pressomancy.object_classes.quadriplex_class.Quartet`, where we specify that the quartets should be created in ``type='solid'`` mode. This configuration is then used to initialize a list of quartets, which are subsequently stored in :class:`~pressomancy.simulation.Simulation`. Storing objects is important because it is the mechanism by which the main simulation object, in a sense the mastermind object in pressomancy, keeps track of all objects, types, and a host of details that help you avoid conflicts and bugs.

Quartet objects are then grouped three at a time in an iterable. We do this because we want to use them to make :class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` objects, which expect exactly three associated quartets. This is where hierarchy really matters. Once again, we create a configuration for quadriplexes, where we specify the bonding mode and pass the bond handle, created beforehand, that will be used during object creation to bond the quartets together. Note how configurations are always object-specific in pressomancy. In fact, simulation-object configurations in pressomancy are part of the object definition and are strict and checked. We also specify the size of the quadriplex, which is important for later placement. The size parameter tells pressomancy how much space it needs to reserve to ensure that, once an object is placed, there are no possible overlaps with other objects in the simulation box. The quadriplexes are then initialized and stored in the simulation object.

This becomes a repeatable process whenever you build a hierarchy of objects in pressomancy: create a configuration, initialize the objects with that configuration, and store them in the simulation class. Since we want to create G4 multimers, which are essentially polymers with a particular type of monomer, we use :class:`~pressomancy.object_classes.filament_class.Filament` as the top-level object in the hierarchy. As before, we need to group quadriplexes into filaments. The grouping is again done with an iterable, and the filament configuration is created with the appropriate size, spacing, and number of parts. You can always see which parameters are required to configure a simulation object in pressomancy, because they are explicitly stated in the class definition. The filament size is calculated from the size of the quadriplexes and the spacing between them, as the diameter of a circumscribed sphere around a G4 multimer. The filament objects are then initialized and stored in the simulation. With this, the top-down hierarchy of objects needed to make a G4 multimer is defined and configured, and the simulation is aware of all the objects and their relationships.

Next, we call ``sim.set_objects(filaments)`` to place the filaments in the simulation box. This is a crucial step because it assigns spatial context to the filaments and their children. The :class:`~pressomancy.object_classes.filament_class.Filament` configuration uses ``n_parts`` and ``spacing`` to define the local chain geometry that its ``build_function`` should generate when the simulation assigns a top-level volume. At that point, one call to ``set_objects(filaments)`` is enough to materialize the whole G4 multimer suspension recursively.

That is the key modeling shift. The script does not manually calculate global quartet coordinates or micromanage each quadriplex placement. It prepares the hierarchy, registers it with the simulation, and lets the top-level object class drive construction downward.

Finally, the last step is to call :meth:`~pressomancy.object_classes.filament_class.Filament.bond_quadriplexes` to place the bonds that were passed in the quadriplex configuration. This is a common pattern in pressomancy: placement and bonding are often separate steps, especially when the bonding depends on the spatial arrangement of the objects.

.. code-block:: python

   part_per_filament = 8
   n_filaments = 4
   n_quartets = 3 * part_per_filament * n_filaments

   quartet_cfg = Quartet.config.specify(
       espresso_handle=sim.sys,
       type='solid',
   )
   quartets = [Quartet(config=quartet_cfg) for _ in range(n_quartets)]
   sim.store_objects(quartets)

   quadriplex_bond = BondWrapper(
       espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2 * 1.5)
   )
   filament_bond = BondWrapper(
       espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2 * 1.5)
   )
   grouped_quartets = [quartets[i:i + 3] for i in range(0, len(quartets), 3)]
   quadriplex_cfgs = [
       Quadriplex.config.specify(
           size=np.sqrt(3) * 5.,
           espresso_handle=sim.sys,
           bond_handle=quadriplex_bond,
           associated_objects=group,
       )
       for group in grouped_quartets
   ]
   quadriplexes = [Quadriplex(config=cfg) for cfg in quadriplex_cfgs]
   sim.store_objects(quadriplexes)

   grouped_quadriplexes = [
       quadriplexes[i:i + part_per_filament]
       for i in range(0, len(quadriplexes), part_per_filament)
   ]
   filament_cfgs = [
       Filament.config.specify(
           size=quadriplexes[0].params['size'] * part_per_filament
                + np.sqrt(3) * filament_bond.r_0 + (part_per_filament - 1),
           n_parts=part_per_filament,
           espresso_handle=sim.sys,
           bond_handle=filament_bond,
           associated_objects=group,
           spacing=6.,
       )
       for group in grouped_quadriplexes
   ]
   filaments = [Filament(config=cfg) for cfg in filament_cfgs]
   sim.store_objects(filaments)
   sim.set_objects(filaments)

   for filament in filaments:
       filament.bond_quadriplexes()

Broken Quartets And Telomeres
-----------------------------

The folded telomeric example starts from the same quartet -> quadriplex logic,
but it changes both the quartet chemistry and the top-level object. Quartets
are created in ``type='broken'`` mode, which requires an explicit
``bond_handle`` and changes the internal typing pattern used during quartet
construction. In broken mode, selected particles are promoted or reassigned to
roles such as ``real``, ``circ``, ``squareA``, and ``squareB`` so that the
later fold logic has the expected local topology available.

Those broken quartets are still grouped into quadriplexes. The real change
comes one level higher, where the top-level object becomes
:class:`~pressomancy.object_classes.tel_sequence.TelSeq`. Like
:class:`~pressomancy.object_classes.filament_class.Filament`, it defines a
chain-like ``build_function`` so the simulation can place complete top-level
units in the box. After placement,
:meth:`~pressomancy.object_classes.tel_sequence.TelSeq.wrap_into_Tel`
interprets the requested fold type and adds the corresponding diagonal and
across bonds between corners.

The consequence is important. The folded example does not need a new global
placement algorithm. It needs a different top-level object whose local
post-placement logic is more specific.

.. code-block:: python

   part_per_filament = 8
   fold_types = ['parallel', 'hybrid', 'antiparallel']
   n_telomeres = len(fold_types)
   n_quartets = 3 * part_per_filament * n_telomeres

   quartet_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=1.5))
   quadriplex_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.0))
   diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1.0, d_r_max=1.5))
   across_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1.0, d_r_max=1.5))
   quartet_cfg = Quartet.config.specify(
       bond_handle=quartet_bond,
       type='broken',
       espresso_handle=sim.sys,
   )
   quartets = [Quartet(config=quartet_cfg) for _ in range(n_quartets)]
   sim.store_objects(quartets)

   grouped_quartets = [quartets[i:i + 3] for i in range(0, len(quartets), 3)]
   quadriplex_cfgs = [
       Quadriplex.config.specify(
           associated_objects=group,
           espresso_handle=sim.sys,
           bonding_mode='ftf',
           bond_handle=quadriplex_bond,
           size=np.sqrt(3) * 5,
       )
       for group in grouped_quartets
   ]
   quadriplexes = [Quadriplex(config=cfg) for cfg in quadriplex_cfgs]
   sim.store_objects(quadriplexes)

   grouped_quadriplexes = [
       quadriplexes[i:i + part_per_filament]
       for i in range(0, len(quadriplexes), part_per_filament)
   ]
   telseq_cfgs = [
       TelSeq.config.specify(
           n_parts=part_per_filament,
           espresso_handle=sim.sys,
           associated_objects=grouped_quadriplexes[idx],
           size=quadriplexes[0].params['size'] * part_per_filament
                + np.sqrt(3) * quadriplex_bond.r_0 + (part_per_filament - 1),
           bond_handle=quadriplex_bond,
           diag_bond_handle=diag_bond,
           across_bond_handle=across_bond,
           spacing=6.,
           type=fold_type,
       )
       for idx, fold_type in enumerate(fold_types)
   ]
   telomeres = [TelSeq(config=cfg) for cfg in telseq_cfgs]
   sim.store_objects(telomeres)
   sim.set_objects(telomeres)

   for telomere in telomeres:
       telomere.wrap_into_Tel()

Key Options
-----------

.. rubric:: Quartet

The first consequential choice is the quartet ``type``. ``solid`` gives a
rigid-body style quartet with one real center and virtual sites. ``broken``
gives a more chemically differentiated internal layout and is the mode used by
the telomeric sample. Broken quartets require ``bond_handle`` to be set at
construction time, so the choice is structural rather than cosmetic.

.. rubric:: Quadriplex

For quadriplexes, the key option is ``bonding_mode``. ``ftf`` is the default
and bonds compatible corner particles across quartets. ``ctc`` instead bonds
the central quartet to the top and bottom quartet through their main real
particles. This choice changes the bonded architecture and also affects later
mechanics such as
:meth:`~pressomancy.object_classes.quadriplex_class.Quadriplex.add_bending_potential`.

.. rubric:: Filament

For filaments, the practical parameters are ``n_parts``, ``spacing``, and
``size``. ``n_parts`` must match the number of associated quadriplexes in a
composite filament. ``spacing`` controls the chain geometry produced by the
filament ``build_function``. ``size`` is what the simulation sees when it
decides whether whole filaments can be placed without overlap.

.. rubric:: TelSeq

:class:`~pressomancy.object_classes.tel_sequence.TelSeq` adds the top-level
fold ``type``. On the current branch, the supported values are ``parallel``,
``hybrid``, and ``antiparallel``. These do not alter the global placement
stage. They alter the local post-placement bond pattern applied by
:meth:`~pressomancy.object_classes.tel_sequence.TelSeq.wrap_into_Tel`.

Common Mistakes
---------------

The most common failure in the solid workflow is a grouping mismatch.
Quadriplex construction assumes groups of exactly three quartets, and a
composite filament assumes ``n_parts == len(associated_objects)``. If the last
group in a sequence is shorter than expected, the visible failure often occurs
well after the original mistake, so it is worth validating group lengths
before configuration objects are created.

The second recurring pitfall is to conflate placement with higher-level
bonding. In these examples they are intentionally separate steps.
``set_objects`` creates the hierarchy and gives every object a spatial context.
Methods such as
:meth:`~pressomancy.object_classes.filament_class.Filament.bond_quadriplexes`
and :meth:`~pressomancy.object_classes.tel_sequence.TelSeq.wrap_into_Tel`
then add the larger-scale couplings that depend on the already created
geometry.

Finally, ``ctc`` and ``ftf`` should be treated as genuinely different modeling
choices. They produce different bonded structures, and downstream mechanics
should be written with that choice in mind rather than assuming the two modes
are interchangeable.

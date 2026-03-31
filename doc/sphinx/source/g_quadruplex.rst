G-quadruplex Assembly
=====================

This tutorial is where the general ideas from the user guide are pushed into a
realistic object hierarchy. It is not a separate conceptual layer. It is a
worked example that shows why pressomancy builds from top-level objects,
propagates construction downward, and keeps ownership and connectivity in the
IO output rather than only particle coordinates. On the current branch, the
core build path is
:class:`~pressomancy.object_classes.quadriplex_class.Quartet` ->
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` ->
:class:`~pressomancy.object_classes.filament_class.Filament` or
:class:`~pressomancy.object_classes.tel_sequence.TelSeq`.

The simulation never places quartets directly into the global box. It places a
top-level object and lets that object recursively build the lower levels
beneath it. The two sample workflows on ``main`` make that design concrete: a
poly-G4 filament built from solid quartets, and a folded telomeric sequence
built from broken quartets with additional top-level fold logic.

What Gets Built
---------------

In ``samples/poly_BRACO.py``, quartets are the rigid building blocks. Three
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
:class:`~pressomancy.object_classes.tel_sequence.TelSeq` replaces the plain
filament and adds fold-specific post-placement logic. Quartets and
quadriplexes are still built in the same general top-down style, but the final
local couplings depend on the chosen telomeric fold type.

Solid Quartets
--------------

The solid workflow begins by constructing quartets in ``type='solid'`` mode,
storing them, and grouping them three at a time into quadriplexes. This is the
first point where hierarchy really matters.
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` expects
exactly three associated quartets. Its
:meth:`~pressomancy.object_classes.quadriplex_class.Quadriplex.set_object`
method then places those quartets at ``pos``, ``pos + r_0 * ori``, and
``pos - r_0 * ori`` before applying the selected bonding mode.

Once quadriplexes have been prepared, they are grouped into filaments. The
:class:`~pressomancy.object_classes.filament_class.Filament` configuration
uses ``n_parts`` and ``spacing`` to define the local chain geometry that its
``build_function`` should generate when the simulation assigns a top-level
volume. At that point, one call to ``set_objects(filaments)`` is enough to
materialize the whole poly-G4 assembly recursively.

That is the key modeling shift. The script does not manually calculate global
quartet coordinates or micromanage each quadriplex placement. It prepares the
hierarchy, registers it with the simulation, and lets the top-level object
class drive construction downward.

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

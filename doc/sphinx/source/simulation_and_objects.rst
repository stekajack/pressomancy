Simulation And Objects
======================

Pressomancy is organized around one hard separation of responsibilities. The
:class:`~pressomancy.simulation.Simulation` class owns the global system: the
ESPResSo handle, the box, the registered top-level objects, and the shared
particle-type bookkeeping. Simulation objects own everything local: what they
are made of, how much space they require, and how they should materialize once
a position and orientation have been assigned. This chapter explains why that
split exists and what it means when you author a new object class.

Why The Split Matters
---------------------

The simulation box is a shared resource, so it must be managed in one place.
That is why :class:`~pressomancy.simulation.Simulation` is responsible for
placement. An individual object class does not decide where it sits globally.
Instead, it declares the information the simulation needs in order to place it
sensibly: its size, its required features, and the local construction routine
that should be used once a placement target has been chosen.

The object then takes over. After the simulation has assigned a position and
orientation, the object decides what particles should be created, how those
particles should be typed, which bonds should exist locally, and whether child
objects should be placed recursively beneath it. In other words, the
simulation answers "where does this top-level thing go?" and the object
answers "what exactly gets built there?"

That distinction becomes essential once objects are hierarchical.
:class:`~pressomancy.object_classes.filament_class.Filament` may own
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` children,
which may in turn own
:class:`~pressomancy.object_classes.quadriplex_class.Quartet` children. The
simulation places only the filament. The filament then places its
quadriplexes, and each quadriplex places its quartets. Pressomancy therefore
builds from the top down even though the actual particles are created at the
leaves.

The placement pipeline follows that logic directly.
:meth:`~pressomancy.simulation.Simulation.set_objects` calls
:func:`~pressomancy.helper_functions.partition_cubic_volume` to generate
candidate regions in the box. The helper starts from an FCC lattice because it
provides a dense and regular set of trial centers. For each accepted center,
the simulation calls the object's ``build_function`` to ask what local point
pattern belongs inside that allocated region. The result is then handed to
:meth:`~pressomancy.object_classes.object_class.Simulation_Object.set_object`,
which is where the real ESPResSo particles finally appear.

What A Simulation Object Is
---------------------------

The common contract for all object classes is enforced by the metaclass
:class:`~pressomancy.object_classes.object_class.Simulation_Object`. That is
important not just because it standardises class shape, but because the entire
library depends on that consistency as a prerequisite for interoperability.
If different objects did not obey the same bookkeeping rules, the simulation
manager could not safely store them, place them, recurse through them, or
write their ownership structure to disk in a uniform way.

At class definition time, the metaclass expects the object family to declare
the metadata the simulation needs in order to reason about it:
``required_features``, ``numInstances``, ``part_types``, ``simulation_type``,
and ``config``. At instance level, the object is expected to expose
``who_am_i``, ``type_part_dict``, ``associated_objects``, and ``sys``. Those
requirements are not there for style. They are what lets a
:class:`~pressomancy.object_classes.part_class.GenericPart`, a
:class:`~pressomancy.object_classes.filament_class.Filament`, and a
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` all be
handled by the same :class:`~pressomancy.simulation.Simulation` machinery.

The metaclass also operationalises the worker functions that keep the peace
across the object library. It injects shared methods such as
:meth:`~pressomancy.object_classes.object_class.Simulation_Object.add_particle`,
:meth:`~pressomancy.object_classes.object_class.Simulation_Object.change_part_type`,
:meth:`~pressomancy.object_classes.object_class.Simulation_Object.get_owned_part`,
:meth:`~pressomancy.object_classes.object_class.Simulation_Object.bond_owned_part_pair`,
and :meth:`~pressomancy.object_classes.object_class.Simulation_Object.delete_owned_parts`.
Those methods are not merely conveniences. They are the routes through which
pressomancy keeps track of ownership chains, part-type assignments, and the
relationship between object-local state and simulation-level state.

This is why direct ad hoc manipulation can become dangerous. You can always
reach into ESPResSo directly, take a particle handle, and change its type by
hand. Pressomancy cannot stop you. But if you do that outside the library's
worker functions, pressomancy may no longer know what happened. The local
``type_part_dict`` may become inconsistent, the simulation-level
``part_types`` registry may no longer reflect reality, and later logic that
relies on shared bookkeeping can quietly drift out of sync. In practice, the
metaclass exists to make that sort of drift less likely.

A particularly non-obvious part of this arrangement is
:meth:`~pressomancy.simulation.Simulation.modify_system_attribute`. During
:meth:`~pressomancy.simulation.Simulation.store_objects`, objects are wired to
that simulation-side method so that controlled changes to shared attributes can
be propagated through the library without the user having to manage every piece
by hand. This is one of the more hidden parts of pressomancy, but it is also
one of the reasons the object library can remain internally consistent while
still allowing object-local methods to update simulation-level bookkeeping.

One detail matters more than it may first appear to. Every simulation object
participates in the same build pipeline through ``build_function``. If a class
does not define one explicitly, the metaclass supplies a default
:class:`~pressomancy.helper_functions.RoutineWithArgs` instance. For simple
single-particle objects, that may be enough. For chain-like or composite
objects such as
:class:`~pressomancy.object_classes.filament_class.Filament` and
:class:`~pressomancy.object_classes.tel_sequence.TelSeq`, ``build_function``
is usually configured in ``__init__`` so that the simulation knows how many
local positions to request, how far apart they should be, and what geometric
constraints should apply during placement.

The class-level ``config`` object is the other half of the contract.
:class:`~pressomancy.object_classes.object_class.ObjectConfigParams` provides a
strict configuration template for each object family. Common keys such as
``espresso_handle``, ``associated_objects``, ``size``, and ``n_parts`` recur
across the library, while each class may extend that base with its own
parameters. ``config.specify(...)`` is the intended way to produce
instance-specific configurations. It is deliberately strict: you may override
known keys, but you may not quietly invent new ones at call site.

Extension Patterns
------------------

The most important point here is that the metaclass is the primary extension
manager in pressomancy. If you want a new object to participate cleanly in the
library, the first question is not "what should I inherit from?" but rather
"what contract must this object satisfy so that it remains interoperable with
all the others?" Inheritance is secondary to that. It is useful when an object
really is a specialisation of an existing implementation, but the metaclass is
what keeps the overall ecosystem coherent.

In practice, there are three useful ways to extend the library. The first, and
in many ways the most direct, is to use ``metaclass=Simulation_Object``
explicitly. This is the right route when the object is really a manager of
other objects, or when its construction logic is distinctive enough that you do
not gain much by pretending it is just a special case of an existing base.
That is the world of
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex`,
:class:`~pressomancy.object_classes.filament_class.Filament`,
:class:`~pressomancy.object_classes.tel_sequence.TelSeq`, and
:class:`~pressomancy.object_classes.otp_molecule_class.OTP`.

The second route is ordinary inheritance from an existing primitive base such
as :class:`~pressomancy.object_classes.part_class.GenericPart` or
:class:`~pressomancy.object_classes.rigid_obj.GenericRigidObj`. This is the
right choice when a new object genuinely adds or changes behaviour. For
example, :class:`~pressomancy.object_classes.egg_model_part.EGGPart` is not
just a renamed primitive. It changes required features, part-type structure,
and object setup in a substantive way. Likewise,
:class:`~pressomancy.object_classes.multicore_particle.MulticorePart` extends
rigid-object behaviour with additional magnetic functionality.

The third route is lightweight aliasing, and this is important because it is
what stops the object library from ballooning for trivial variations. Very
often you do not need a conceptually new implementation. You need a mnemonic
specialisation of an existing primitive or rigid object, with different
metadata, a different preset resource, or a clearer semantic identity in a
script. In that case, a thin alias-style class is enough. The rigid-object
side makes this especially explicit through ``config['alias']``, which selects
``resources/<alias>.txt``. Classes such as
:class:`~pressomancy.object_classes.raspberry_sphere.RaspberrySphere` and
:class:`~pressomancy.object_classes.multicore_particle.MulticorePart` largely
reuse :class:`~pressomancy.object_classes.rigid_obj.GenericRigidObj` and pin a
particular alias or add only a little extra behaviour. The same general idea
applies more broadly: if the difference is mostly one of identity, preset
configuration, or mnemonic clarity, a lightweight alias is preferable to a new
heavy implementation.

Put differently: use the metaclass contract first, inheritance when behaviour
really changes, and alias-style specialisation when what you mostly need is a
stable name and a preset shape inside the library.

Feature requirements fit naturally into the same story. Object-level checks
come from ``required_features`` and are enforced during
:meth:`~pressomancy.simulation.Simulation.store_objects`. Method-level checks
happen inside the relevant methods. On the current branch,
:class:`~pressomancy.object_classes.egg_model_part.EGGPart` requires
``EGG_MODEL``,
:class:`~pressomancy.object_classes.stoner_wohlfarth_part.SWPart` requires
``THERMAL_STONER_WOHLFARTH``, and classes derived from
:class:`~pressomancy.object_classes.rigid_obj.GenericRigidObj` require
``VIRTUAL_SITES_RELATIVE``.

Minimal Template
----------------

The following skeleton shows the smallest direct use of the metaclass. It is
not a universal recipe, but it does capture the contract that every custom
object class must satisfy.

.. code-block:: python

   from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
   from pressomancy.helper_functions import PartDictSafe, SinglePairDict

   class MyObject(metaclass=Simulation_Object):
       required_features = []
       numInstances = 0
       simulation_type = SinglePairDict("my_object", 123)
       part_types = PartDictSafe({"real": 1})
       config = ObjectConfigParams(my_param=1.0)

       def __init__(self, config: ObjectConfigParams):
           self.sys = config["espresso_handle"]
           self.params = config
           self.associated_objects = config["associated_objects"]
           self.who_am_i = MyObject.numInstances
           MyObject.numInstances += 1
           self.type_part_dict = PartDictSafe({"real": []})

       def set_object(self, pos, ori):
           part = self.add_particle(type_name="real", pos=pos, rotation=(True, True, True))
           part.director = ori
           return self

For a true container object, ``set_object`` usually should not create the
entire structure directly. It should translate the received placement context
into local placements for its ``associated_objects`` and delegate to them.
That is how
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex`,
:class:`~pressomancy.object_classes.filament_class.Filament`, and
:class:`~pressomancy.object_classes.tel_sequence.TelSeq` work.

One practical rule matters more than the template itself: register objects
before you place them.
:meth:`~pressomancy.simulation.Simulation.store_objects` recursively stores
child objects, binds ``modify_system_attribute``, and updates the global
particle-type bookkeeping. Without that step, the object may exist as a Python
instance, but it is not yet integrated into the shared simulation workflow.

Common Mistakes
---------------

The most common authoring mistake is to treat ``size`` and ``build_function``
as descriptive extras. They are not. ``size`` is the object's statement about
how much global room it occupies. ``build_function`` is the object's statement
about what local arrangement should be generated within that room. If either
one is wrong, placement problems will surface later and often in misleading
ways.

The second common mistake is to treat ``associated_objects`` as passive data.
In composite classes, child objects are part of the construction logic itself.
A robust ``set_object`` implementation should therefore be explicit about the
assumptions it makes: exact child counts, expected object types, required
part-type keys, and any geometry-specific invariants. The existing composite
classes are useful templates precisely because they encode those assumptions
openly instead of relying on informal convention.

If you want to see the same ideas applied to a concrete scientific example,
continue to :doc:`g_quadruplex`.

.. _Introduction:

Introduction
============

Pressomancy is a package aimed at providing a general framework for the
systematic construction of simulation objects in the abstract sense.
Functionally, it is a library of complex simulation objects that can be used
with `EspressoMD <https://espressomd.github.io/doc/>`_ for molecular
dynamics simulations, together with a collection of tools and procedures that
are repeatedly useful in MD work.

This project started as a way to consolidate some of my own legacy code
alongside the best practices I had adopted at the time. Initially, that was a
frustrating and error-prone exercise. To make it survivable, I ended up
introducing a kind of babysitter mechanism to catch recurring mistakes and
adding tests for every feature before actually implementing it, because I tend
to be wary of regressions. Over time, that effort started to look less like a
private cleanup job and more like the skeleton of an actual package. That is
how Pressomancy came about.

There are many simulation packages and many ways to perform simulations.
However, at their core they all revolve around simulating some system, or some
combination of systems. My experience has been that despite working on very
different problems over the years, simulation scripts often end up looking
remarkably similar, and I repeatedly need the same kinds of tools across
projects. That naturally leads to the question: why not define a
:class:`~pressomancy.simulation.Simulation` that lets me replicate the overall
structure of an MD simulation and keep track of the general elements of
interest? And, related to that, why not define a broad developer guide for a
simulation object: a structure flexible enough to accommodate arbitrary
objects, but disciplined enough that those objects can interoperate with the
:class:`~pressomancy.simulation.Simulation` and with one another, present or
future? If that structure is chosen carefully, quite a lot of useful
functionality then comes for free.

With some effort, that skeleton now exists. It is there to support the growth
of a library of simulation objects with arbitrary complexity, which can be
managed and combined in MD simulations in a scalable and reasonably safe way.
The underlying principle remains a strict separation of ownership and
responsibility. The :class:`~pressomancy.simulation.Simulation` owns what
properly belongs to the simulation: the global state, the box, the placement
logic, and the IO context. Simulation objects own what properly belongs to the
object: what it is, how much space it needs, how it makes itself, what it
owns, and how its children should be instantiated beneath it.

As it stands, Pressomancy is designed as a wrapper around an EspressoMD
instance. In practice, it must be run with the Espresso interpreter to
function, and for now that satisfies all my needs. Looking ahead, I would like
the library eventually to become fully decoupled from any specific simulation
package, with API bindings for supported backends.

This is also why a number of the package's design choices are not arbitrary.
Placement starts from the top-level object. Composite objects are built
recursively. IO stores not just particle properties but also contextual
relationships. Analysis can then work in terms of objects and their
connectivity rather than only in terms of raw particle arrays. The practical
benefit is straightforward: larger structured systems become easier to build,
easier to inspect, and much easier to continue after the fact.

How To Use This Guide
---------------------

This guide is written as a workflow, not as a catalog. The fastest route in
is :doc:`first_simulation`, which walks through one complete run: define an
object family, store it, place it, write one HDF5 frame, and inspect the
result. After that, :doc:`simulation_and_objects` explains why the build
pipeline is arranged the way it is and what it means to define a new object
class. :doc:`io_and_analysis` then shows how the HDF5 layout supports both
analysis and source-driven initialization. The separate tutorials section then
applies the same ideas to a more substantial scientific example in
:doc:`g_quadruplex`.

The package is still pre-release. The overall design is already coherent
enough to learn and use, but details may still move between releases. When
you need exact signatures or exhaustive class listings, use the API reference
available under :doc:`modules`. The narrative chapters are meant to explain
how the pieces fit together and in what order they are usually used.

Repository Layout
-----------------

At repository level, pressomancy separates the simulation manager, the object
library, analysis helpers, and examples:

.. code-block:: text

    pressomancy/
    ├─ pressomancy/
    │  ├─ simulation.py
    │  ├─ helper_functions.py
    │  ├─ analysis/
    │  ├─ object_classes/
    │  └─ resources/
    ├─ doc/sphinx/
    ├─ samples/
    └─ test/

In normal use, three areas matter most. :mod:`pressomancy.simulation`
contains the top-level orchestration logic. :mod:`pressomancy.object_classes`
contains the reusable object families that plug into that orchestration.
:mod:`pressomancy.analysis.data_analysis` contains the HDF5-backed analysis
helpers that make the saved output convenient to inspect. The scripts in
``samples/`` are therefore more than demonstrations. They are the clearest
record of how the intended build patterns are supposed to be used in practice.

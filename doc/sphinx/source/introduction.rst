.. _Introduction:

Introduction
============

Pressomancy is a package aimed at providing a general framework to systematically build simulation objects (abstract). Functionally, it is a library of complex simulation objects that can be used in the EspressoMD simulation package for Molecular Dynamics (MD) simulations. It also includes various helpful tools and procedures commonly used in MD simulations.

This project started as a way to consolidate some of my own legacy code alongside best practices I had adopted at the time. Initially, this was a frustrating and error-prone endeavor. To address this, I implemented a kind of “babysitter mechanism” to prevent recurring errors and added tests for every feature before actually implementing it, as I tend to struggle with anticipating regressions. Over time, this effort started to resemble something that could be a package—and that’s how Pressomancy came to be.

There are many different simulation packages and ways to perform simulations. However, at their core, all of them revolve around simulating some system or combinations thereof. My experience has shown that despite working on varied systems over the years, simulation scripts often end up looking very similar, and I frequently need the same tools across different projects. So, why not create a Simulation that allows me to replicate the overall structure and track the general elements of interest in an MD simulat ion? Additionally, why not provide a Simulation Object developer guide—something that defines a broad structure to fit arbitrary simulation objects into a standardized framework? Such a guide would encourage developers to make design choices that enable interoperability with both the Simulation and any other objects, current or future. This approach also provides some functionality for free.

With some effort, this skeleton now exists to support the growth of a library of simulation objects with arbitrary complexity. These objects can be managed and combined in MD simulations in a scalable and safe way.

As it stands, Pressomancy is designed as a wrapper for an EspressoMD instance. In fact, it must be run with the Espresso interpreter to function. This setup currently meets all my needs. Looking ahead, I would like the library to eventually become fully decoupled from any specific simulation package, with API bindings for supported packages. 

.. _Basic program structure:

Basic program structure
-----------------------

Below is a schematic depiction of the project's file hierarchy:

.. code-block:: text

    pressomancy/
    ├─ pressomancy/
    │  ├─ simulation.py
    │  ├─ helper_functions.py
    │  ├─ object_classes/
    │  │  ├─ __init__.py
    │  │  ├─ object_class.py
    │  │  ├─ ...
    │  │  └─ 
    │  └─ __init__.py
    ├─ doc/
    │  ├─ sphinx/
    │  │  ├─ ...
    │  │  └─ 
    │  └─ ...
    ├─ samples/
    │  ├─ ...
    │  └─ 
    └─ test/
       ├─ ...
       └─ 


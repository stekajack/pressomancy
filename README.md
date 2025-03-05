
# Pressomancy

[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue.svg)](https://stekajack.github.io/pressomancy/)
[![CI Pipeline](https://github.com/stekajack/pressomancy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/stekajack/pressomancy/actions/workflows/ci.yml)

<table>
  <tr>
    <td>
      <strong>Pressomancy</strong> is a Python framework built on top of 
      <a href="http://espressomd.org/">EspressoMD</a>, designed to streamline and abstract 
      the management of objects in molecular simulations. By creating a structured 
      environment for simulation objects, <strong>Pressomancy</strong> enables easy integration, 
      customization, and scalability for complex simulations with minimal setup.
      <br><br>
      The framework provides an abstract, modular approach for defining simulation components, 
      ensuring compliance and ease of integration across various objects, such as magnetic 
      filaments, G4 multimers, ligands, and more.
    </td>
   <td><img src="doc/logo/pressomancy_logo_v1.png" width="800"></td>
  </tr>
</table>

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
---

## Features

- **Abstract Framework**: Build and manage simulations with a consistent, object-oriented structure.
- **Custom Simulation Objects**: Create custom simulation objects with compatibility and compliance.
- **Simulation Management**: Centralized, singleton-like simulation class to wrap and control EspressoMD functionality with separate lifetime and memory management.
- **Extendable**: Easily extend the framework with new simulation objects by adhering to the abstract SimulationObject metaclass.
- **Pre-Built Objects**: Includes ready-to-use simulation objects such as magnetic filaments, G4 multimers, and ligands.

## Installation

To install **Pressomancy**, use pip ( -e flag for edit mode):
```bash
pip install pressomancy
```
To test **Pressomancy**, use:
```bash
$path_to_espresso/build/pypresso -m unittest discover -s test
```

### Requirements
- Python 3.x
- EspressoMD

## Quick Start

Here’s a quick example to get started with **Pressomancy**.

1. **Initialize the Simulation**: The `Simulation` class is a singleton-like wrapper for managing an EspressoMD instance with specialised methods to manage a molecular simualation.
2. **Add Simulation Objects**: Add various simulation objects (e.g., magnetic filaments, multimers) that inherit from the `SimulationObject` abstract class, ensuring easy integration and compliance.

```python
from pressomancy import Simulation
from pressomancy.objects import Filament

# Initialize EsoressoMD instace and set system parameters
sim_inst = Simulation(box_dim=(10,10,10))
sim_inst.set_sys()
# Make a custom configuration (object aware)
configuration=Filament.config.specify(params....)
# Add simulation objects
filaments = [Filament(configuration) for x in range(#)]
# Register objects  in the simulation instance
sim_inst.store_objects(filaments)
# Create objects inside the EspressoMD instance
sim_inst.set_objects(filaments)
# Set interactions
sim_inst.set_vdW(key=('type_key',),lj_eps=#)
# Do work
.
.
.
```
## Usage

**Pressomancy** is designed to be extendable and intuitive, with the aim to simplify and systematise complex architecture creation in molecular simulations. Each component in **Pressomancy** adheres to a standardized interface, making it straightforward to add or modify simulation objects.

### Simulation Class

The `Simulation` class manages the EspressoMD instance and provides methods for simulation management (i.e. I/O, properties), ensuring all objects are added and side effects are synchronized.

```python
from pressomancy import Simulation
sim = Simulation() 
```

### Simulation Objects

Objects in **Pressomancy** use the `SimulationObject` metaclass, ensuring that each object is compliant with the framework's structure. Examples include:
- **Filament**: Represents a linear array of objects.
- **Quadriplex**: Non-canonical DNA conformation.
- **SWPart**: Magnetic nanoparticle with internal anisotropy and magnetodynamis via the tSW model.

`SimulationObject` is implemented to facititate contributors adding their own simualtion object. The metaclass adds various methods, hooks and traps to guarantee compatibility and integration with the **Pressomancy** framework..


```python

from pressomancy.object_classes.object_class import Simulation_Object 
class CustomObject(metaclass=Simulation_Object):
    # all required parameter and methods are attached
    # various traps with detailed debugging info
    # get boolean opeation, iteraton context, destructors
    # the rest is up to you
```

## Modules

- **`objects`**: Library of simulation objects.
- **`resources`**: resources and metadata for objects.
- **`analysis`**: Library of utilities and anaysis routines.
- **`simulation`**: Centralized management of the simulation state.

## Contributing

Contributions to **Pressomancy** are welcome! To contribute, please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss your ideas.


# Pressomancy

**Pressomancy** is a Python framework built on top of [EspressoMD](http://espressomd.org/), designed to streamline and abstract the management of molecular simulations. By creating a structured environment for simulation objects, **Pressomancy** enables easy integration, customization, and scalability for complex simulations with minimal setup. 

The framework provides an abstract, modular approach for defining simulation components, ensuring compliance and ease of integration across various objects, such as magnetic filaments, G4 multimers, ligands, and more.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Abstract Framework**: Build and manage simulations with a consistent, object-oriented structure.
- **Custom Simulation Objects**: Create custom simulation objects with guaranteed compatibility and compliance.
- **Singleton Simulation Management**: Centralized, singleton-based simulation class to wrap and control EspressoMD functionality.
- **Extendable**: Easily extend the framework with new simulation objects by adhering to the abstract base classes.
- **Pre-Built Objects**: Includes ready-to-use simulation objects such as magnetic filaments, G4 multimers, and ligands.

## Installation

To install **Pressomancy**, use pip :
```bash
pip install pressomancy
```

### Requirements
- Python 3.x
- EspressoMD

## Quick Start

Here’s a quick example to get started with **Pressomancy**.

1. **Initialize the Simulation**: The `Simulation` class serves as a singleton wrapper for managing EspressoMD simulations.
2. **Add Simulation Objects**: Add various simulation objects (e.g., magnetic filaments, multimers) that inherit from the `SimulationObject` abstract class, ensuring easy integration and compliance.

```python
from pressomancy import Simulation
from pressomancy.objects import MagneticFilament, G4Multimer

# Initialize the singleton simulation
sim = Simulation()

# Add simulation objects
filament = MagneticFilament(params=...)
multimer = G4Multimer(params=...)

sim.add_object(filament)
sim.add_object(multimer)

# Run the simulation
sim.run()
```

## Usage

**Pressomancy** is designed to be extendable and intuitive, focusing on ease of use for complex simulations. Each component in **Pressomancy** adheres to a standardized interface, making it straightforward to add or modify simulation objects.

### Simulation Class

The `Simulation` class manages the EspressoMD simulation, ensuring all objects are added and synchronized correctly. As a singleton, it provides a single point of access and control over the simulation state.

```python
from pressomancy import Simulation

sim = Simulation()  # Always returns the same instance
```

### Simulation Objects

Objects in **Pressomancy** inherit from the `SimulationObject` abstract base class, ensuring that each object is compliant with the framework's structure. Examples include:
- **MagneticFilament**: Represents a filament with magnetic properties.
- **G4Multimer**: Models a G4 multimer structure.
- **Ligand**: Represents ligands or molecules that can interact with other objects.

You can also create custom objects by inheriting from `SimulationObject`.

```python
from pressomancy.objects import MagneticFilament

filament = MagneticFilament(params=...)
```

### Creating Custom Simulation Objects

To implement a new simulation object, inherit from `SimulationObject` and define the required methods. This guarantees compatibility and integration with the **Pressomancy** framework.

```python
from pressomancy.core import SimulationObject

class CustomObject(SimulationObject):
    def __init__(self, params):
        self.params = params
        # Additional initialization code

    def update(self):
        # Define how this object should be updated during each simulation step
        pass
```

## Modules

- **`core`**: Contains base classes and core utilities for **Pressomancy**.
- **`objects`**: Collection of pre-built simulation objects like `MagneticFilament`, `G4Multimer`, and `Ligand`.
- **`simulation`**: Singleton simulation wrapper class, providing centralized management of the simulation state.

## Contributing

Contributions to **Pressomancy** are welcome! To contribute, please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss your ideas.

## License

**Pressomancy** is licensed under the MIT License. See `LICENSE` for more information.
"""
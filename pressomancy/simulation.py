import espressomd
from espressomd.virtual_sites import VirtualSitesRelative
from espressomd import shapes
import espressomd.polymer
import sys as sysos
import numpy as np
import os
import gzip
import pickle
from itertools import combinations_with_replacement
from pressomancy.object_classes import *
from pressomancy.helper_functions import *
import logging

@ManagedSimulation
class Simulation():
    """
    A singleton class designed to manage and simulate a suspension of objects within the ESPResSo molecular dynamics framework.

    The `Simulation` class encapsulates the ESPResSo system and provides methods to configure the simulation, manage objects, and apply various interactions and constraints. It maintains a dictionary of simulation objects, tracks their properties, and delegates object-specific operations to the appropriate methods. 

    Key features include:
    - Managing particle types and their properties.
    - Configuring interactions like Lennard-Jones (LJ), Weeks-Chandler-Andersen (WCA), and magnetic interactions.
    - Supporting lattice Boltzmann (LB) fluid initialization and boundary setup.
    - Storing, setting, and removing simulation objects.
    - Providing utilities for avoiding instabilities, generating positions, and managing simulation data.

    Attributes:
        no_objects (int): The number of objects currently stored in the simulation.
        objects (list): A list of objects stored in the simulation.
        part_types (PartDictSafe): A dictionary-like object tracking particle types and their associated properties.
        seed (int): A random seed for reproducibility, generated at initialization.
        partitioned (bool): Indicates whether the simulation box is partitioned.
        part_positions (list): A list of particle positions generated for the simulation.
        volume_size (float): The size of the volume assigned to each object.
        volume_centers (list): A list of centers of the partitioned volumes.

    Methods:
        __init__(box_dim):
            Initializes the simulation class with a given box dimension.

        set_sys(timestep, min_global_cut):
            Configures the ESPResSo system's basic parameters, such as periodicity, time step, and cell system properties.

        modify_system_attribute(requester, attribute_name, action):
            Validates and modifies a system attribute if permitted by the object's permissions.

        store_objects(iterable_list):
            Stores simulation objects and updates particle types and attributes.

        set_objects(objects):
            Generates random positions and orientations for managed objects and sets them using object-specific methods.

        unstore_objects(iterable_list):
            Removes specified objects from the simulation and updates relevant attributes.

        delete_objects():
            Deletes all parts owned by stored objects by calling their delete method.

        mark_for_collision_detection(object_type, part_type):
            Marks specific objects for collision detection and prepares them for covalent bond marking.

        init_magnetic_inter(actor_handle):
            Initializes direct summation magnetic interactions in the simulation.

        set_steric(key, wca_eps, sigma):
            Configures WCA interactions between specified particle types.

        set_steric_custom(pairs, wca_eps, sigma):
            Configures custom WCA interactions for specific particle type pairs.

        set_vdW(key, lj_eps, lj_size):
            Sets Lennard-Jones interactions for specified particle types.

        set_vdW_custom(pairs, lj_eps, lj_size):
            Configures custom Lennard-Jones interactions for specific particle type pairs.

        init_lb(kT, agrid, dens, visc, gamma, timestep):
            Initializes a lattice Boltzmann fluid with the specified parameters.

        create_flow_channel(slip_vel):
            Sets up lattice Boltzmann boundaries for a flow channel.

        avoid_explosion(F_TOL, MAX_STEPS, F_incr, I_incr):
            Caps forces iteratively to avoid simulation instabilities due to initial overlaps.

        magnetize(part_list, dip_magnitude, H_ext):
            Applies Langevin magnetization to compute dipole moments of particles.

        set_H_ext(H):
            Configures an external homogeneous magnetic field in the simulation.

        get_H_ext():
            Retrieves the current external homogeneous magnetic field.

        init_pickle_dump(path_to_dump):
            Initializes a pickle file to store simulation data.

        load_pickle_dump(path_to_dump):
            Loads simulation data from a pickle dump file.

        dump_to_init(path_to_dump, dungeon_witch_list, cnt):
            Appends simulation data for a specific timestep to an existing pickle dump.

        generate_positions(min_distance):
            Generates random positions for objects while ensuring a minimum distance between them.

    Notes:
        - The class assumes that the ESPResSo system is already instantiated and wraps the system handle during initialization. The initialisation and lifetime is managed by the decorator class.
        - Many methods rely on specific attributes or methods being implemented in the stored objects. This is why any object that is to be safely used by Simulation should use the SimulationObject metaclass.
        - This class is designed to be extensible for different types of interactions and constraints.
    """
    
    object_permissions=['part_types']
    _sys=espressomd.System
    def __init__(self, box_dim):
        self.no_objects = 0
        self.objects = []
        self.part_types = PartDictSafe({})
        self.seed = int.from_bytes(os.urandom(2), sysos.byteorder)
        self.partitioned=None
        self.part_positions=[]
        self.volume_size=None
        self.volume_centers=[]
        # self.sys=espressomd.System(box_l=box_dim) is added and managed by the singleton decrator!

    def set_sys(self, timestep=0.01, min_global_cut=3.0,have_quaternion=False):
        '''
        Set espresso cellsystem params, and import virtual particle scheme. Run automatically on initialisation of the System class.
        '''
        np.random.seed(seed=self.seed)
        logging.info(f'core.seed: {self.seed}')
        self.sys.periodicity = (True, True, True)
        self.sys.time_step = timestep
        self.sys.cell_system.skin = 0.5
        self.sys.min_global_cut = min_global_cut
        self.sys.virtual_sites = VirtualSitesRelative(have_quaternion=have_quaternion)
        assert type(self.sys.virtual_sites) is VirtualSitesRelative, 'VirtualSitesRelative must be set. If not, anything involving virtual particles will not work correctly, but it might be very hard to figure out why. I have wasted days debugging issues only to remember i commented out this line!!!'
        logging.info(f'System params have been autoset. The values of min_global_cut and skin are not guaranteed to be optimal for your simualtion and should be tuned by hand!!!')

    def modify_system_attribute(self, requester, attribute_name, action):
        """
        Validates and modifies a Simulation attribute if allowed by the permissions.

        :param requester: The object requesting the modification.
        :param attribute_name: str | The name of the attribute to modify.
        :param action: callable | A function that takes the current attribute value as input and modifies it.
        :return: None
        """
        if hasattr(self, attribute_name) and attribute_name in self.object_permissions:
            action(getattr(self,attribute_name))

        else:
            logging.info("Requester does not have permission to modify attributes.")

    def sanity_check(self,object):
        '''
        Method that checks if the object has the required features to be stored in the simulation. If the object has the required features it is stored in the self.objects list.
        '''
        if not all(feature in espressomd.features() for feature in object.required_features):
            raise MissingFeature

    def store_objects(self, iterable_list):
        '''
        Method stores objects in the self.objects dict, if the object has a n_part and part_types attributes,
        and the list of objects passed to the method is commesurate with the system level attribute n_tot_parts.
        Populates the self.part_types attribute with types found in the objects that are stored.
        All objects that are stored should have the same types stored, but this is not checked explicitly
        '''
        temp_dict={}
        for element in iterable_list:
            if element.params['associated_objects'] != None:
                check=all(associated in self.objects for associated in element.params['associated_objects'])
                if not check:
                    self.store_objects(element.params['associated_objects'])
            assert element not in self.objects, "Lists have common elements!"
            self.sanity_check(element)
            element.modify_system_attribute = self.modify_system_attribute
            self.objects.append(element)
            for key, val in element.part_types.items():
                temp_dict[key]=val
            self.no_objects += 1
        self.part_types.update(temp_dict)
        logging.info(f'{iterable_list[0].__class__.__name__}s stored')

    def set_objects(self, objects):
            '''
            Method that genrates random positions and orientations for the managed objects (3 sheets per quadriplex), loops trough the stored Quadriplexes in self.objects and calls Quadriplex specific set_quadriplex() method. Note that the positions generated need not necessarely be in the simulation box so this should only be used with PBC.
            '''
            assert all(isinstance(item, type(objects[0])) for item in objects), "Not all items have the same type!"
            # centeres, polymer_positions = partition_cubic_volume_oriented_rectangles(big_box_dim=self.sys.box_l, num_spheres=len(
            #     filaments), small_box_dim=np.array([filaments[0].sigma, filaments[0].sigma, filaments[0].size]), num_monomers=filaments[0].n_parts)
            # positions= generate_positions(len(objects), self.sys.box_l, 7.)
            if not self.part_positions:
                centeres, positions, orientations = partition_cubic_volume(box_length=self.sys.box_l[0], num_spheres=len(
                objects), sphere_diameter=objects[0].params['size'],routine_per_volume=objects[0].build_function)
                self.volume_centers.extend(centeres)
                self.part_positions.extend(positions)
                self.volume_size=objects[0].params['size']
            else:
                centeres, positions, orientations = partition_cubic_volume(box_length=self.sys.box_l[0], num_spheres=len(
                objects), sphere_diameter=objects[0].params['size'],routine_per_volume=objects[0].build_function)
                res=get_cross_lattice_noninterceting_volumes(centeres,objects[0].params['size'],self.volume_centers,self.part_positions,self.volume_size,self.sys.box_l[0])
                mask=[key for key,val in res.items() if all(val)]
                positions=positions[mask]
                orientations=orientations[mask]

            logic = (object_el.set_object(pos_el, orient_el)
                    for object_el, pos_el, orient_el in zip(objects, positions, orientations))
            while True:
                try:
                    next(logic)
                except StopIteration:
                    logging.info(f'{objects[0].__class__.__name__} set!!!')
                    break


    def mark_for_collision_detection(self, object_type=Quadriplex, part_type=666):
        assert any(isinstance(ele, object_type) for ele in self.objects), "method assumes simulation holds correct type object"

        self.part_types['marked'] = 666
        objects_iter = [ele for ele in self.objects if isinstance(ele, object_type)]
        assert all((hasattr(ob, 'mark_covalent_bonds') and callable(getattr(ob, 'mark_covalent_bonds')))
                   for ob in objects_iter), "method requires that stored objects have mark_covalent_bonds() method"
        for obj_el in objects_iter:
            obj_el.mark_covalent_bonds(part_type=part_type)

    def init_magnetic_inter(self, actor_handle):
        logging.info('direct summation magnetic interactions initiated')
        dds = actor_handle
        self.sys.actors.add(dds)

    def set_steric(self, key=('nonmagn',), wca_eps=1., sigma=1.):
        '''
        Set WCA interactions between particles of types given in the key parameter.
        :param key: tuple of keys from self.part_types | Default only nonmagn WCA
        :param wca_epsilon: float | strength of the steric repulsion.

        :return: None

        Interaction length is allways determined from sigma.
        '''
        logging.info(f'part types available {self.part_types.keys()} ')
        logging.info(f'WCA interactions initiated for keys: {key}')
        for key_el, key_el2 in combinations_with_replacement(key, 2):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]
                                      ].wca.set_params(epsilon=wca_eps, sigma=sigma)

    def set_steric_custom(self, pairs=[(None, None),], wca_eps=[1.,], sigma=[1.,]):
        """
        Configures custom Weeks-Chandler-Andersen (WCA) interactions for specified particle type pairs.

        This method explicitly sets the WCA interaction parameters (epsilon and sigma) for each pair of particle types provided. 
        It ensures that each interaction pair has corresponding epsilon and sigma values.

        :param pairs: list of tuples | List of particle type pairs (keys from `self.part_types`) for which interactions are defined. Defaults to [(None, None)].
        :param wca_eps: list of float | Strength of the WCA repulsion (epsilon) for each pair. Defaults to [1.0].
        :param sigma: list of float | Interaction range (sigma) for each pair. Defaults to [1.0].
        :return: None
        :raises AssertionError: If the lengths of `pairs`, `wca_eps`, and `sigma` do not match.
        """
        assert len(pairs) == len(wca_eps) and len(pairs) == len(
            sigma), 'epsilon and sigma must be specified explicitly for each type pair'
        logging.info('WCA interactions initiated')
        for (key_el, key_el2), eps, sgm in zip(pairs, wca_eps, sigma):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]
                                      ].wca.set_params(epsilon=eps, sigma=sgm)

    def set_vdW(self, key=('nonmagn',), lj_eps=1., lj_size=1.):
        """
        Configures Lennard-Jones (LJ) interactions for specified particle types.

        This method sets the LJ interaction parameters (epsilon and sigma) for particle types listed in the `key` parameter. 
        The interaction cutoff is automatically set to 2.5 times the LJ size (sigma).

        :param key: tuple of str | Particle type keys from `self.part_types` for which interactions are defined. Defaults to ('nonmagn',).
        :param lj_eps: float | Strength of the LJ attraction (epsilon). Defaults to 1.0.
        :param lj_size: float | Interaction range (sigma). Defaults to 1.0.
        :return: None
        """

        lj_cut = 2.5*lj_size
        for key_el, key_el2 in combinations_with_replacement(key, 2):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]].lennard_jones.set_params(
                epsilon=lj_eps, sigma=lj_size, cutoff=lj_cut, shift=0)
        logging.info(f'vdW interactions initiated initiated for keys: {key}')

    def set_vdW_custom(self, pairs=[(None, None),], lj_eps=[1.,], lj_size=[1.,]):
        """
        Custom setter for Lennard-Jones (LJ) interactions between specified particle type pairs.

        This method allows for the explicit definition of LJ interaction parameters (epsilon and sigma) for each pair of particle types in the simulation.

        :param pairs: list of tuples | Each tuple specifies a pair of keys from `self.part_types` for which interactions are defined. Defaults to [(None, None)].
        :param lj_eps: list of float | Strength of the LJ interaction for each pair. Defaults to [1.0].
        :param lj_size: list of float | Interaction range (sigma) for each pair. Defaults to [1.0].
        :return: None

        :raises AssertionError: If the lengths of `pairs`, `lj_eps`, and `lj_size` are not equal.
        """

        assert len(pairs) == len(lj_eps) and len(pairs) == len(
            lj_size), 'epsilon and sigma must be specified explicitly for each type pair'
        for (key_el, key_el2), eps, sgm in zip(pairs, lj_eps, lj_size):
            lj_cut = 1.5*sgm
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]].lennard_jones.set_params(
                epsilon=eps, sigma=sgm, cutoff=lj_cut, shift=0)
        logging.info('vdW interactions initiated!')

    def init_lb(self, kT, agrid, dens, visc, gamma, timestep=0.01):
        """
        Initializes the lattice Boltzmann (LB) fluid for the simulation.

        This method configures an LB fluid using either CPU or GPU resources, depending on availability. It disables the thermostat, initializes particle velocities to zero, and sets the LB fluid parameters. If another active LB actor exists, it removes it before adding the new LB fluid.

        :param kT: float | Thermal energy (temperature) of the LB fluid.
        :param agrid: int | Grid resolution for the LB method.
        :param dens: float | Density of the LB fluid.
        :param visc: float | Viscosity (kinematic) of the LB fluid.
        :param gamma: float | Coupling constant for the thermostat.
        :param timestep: float | Integration time step for the LB simulation. Default is 0.01.
        :return: LBFluid | The configured lattice Boltzmann fluid object.
        """
        self.sys.thermostat.turn_off()
        self.sys.part.all().v = (0, 0, 0)
        param_dict={'kT':kT, 'seed':self.seed, 'agrid':agrid, 'dens':dens, 'visc':visc, 'tau':timestep}
        if 'CUDA' in espressomd.features():
            logging.info('GPU LB method is beeing initiated')

            lbf = espressomd.lb.LBFluidGPU(**param_dict)
        else:
            logging.info('CPU LB method is beeing initiated')

            lbf = espressomd.lb.LBFluid(**param_dict)
        if len(self.sys.actors.active_actors) == 2:
            self.sys.actors.remove(self.sys.actors.active_actors[-1])
        self.sys.actors.add(lbf)
        gamma_MD = gamma
        logging.info(f'gamma_MD: {gamma_MD}')
        self.sys.thermostat.set_lb(
            LB_fluid=lbf, gamma=gamma_MD, seed=self.seed)
        logging.info(f'LBM is set with the params {lbf.get_params()}.')
        return lbf
    

    def create_flow_channel(self, slip_vel=(0, 0, 0)):
        """
        Sets up LB boundaries for a flow channel.

        :param slip_vel: tuple | Velocity of the slip boundary in the format (vx, vy, vz). Default is (0, 0, 0).
        :return: None
        """
        logging.info("Setup LB boundaries.")
        top_wall = shapes.Wall(normal=[1, 0, 0], dist=1) # type: ignore
        bottom_wall = shapes.Wall( # type: ignore
            normal=[-1, 0, 0], dist=-(self.sys.box_l[0] - 1))

        top_boundary = espressomd.lbboundaries.LBBoundary( # type: ignore
            shape=top_wall, velocity=slip_vel)
        bottom_boundary = espressomd.lbboundaries.LBBoundary(shape=bottom_wall) # type: ignore

        self.sys.lbboundaries.add(top_boundary)
        self.sys.lbboundaries.add(bottom_boundary)

    def avoid_explosion(self, F_TOL, MAX_STEPS=10000, F_incr=100, I_incr=100):
        """
        Iteratively caps force to prevent simulation to explode due to overlapps in the initial configuration, adjusting parameters until teh relative force cahnge between try is smaller than F_TOL.

        :param F_TOL: float | Tolerance for relative force change.
        :param MAX_STEPS: int | Maximum number of integration steps. Default is 10000.
        :param F_incr: int | Increment value for the force cap. Default is 100.
        :param I_incr: int | Increment value for integration steps. Default is 100.
        :return: None
        """
        logging.info('iterating with a force cap.')
        self.sys.integrator.run(0)
        while True:
            try:
                old_force = np.max(np.linalg.norm(
                    self.sys.part.all().f, axis=1))
                self.sys.force_cap = F_incr
                self.sys.integrator.run(I_incr)
                force = np.max(np.linalg.norm(self.sys.part.all().f, axis=1))
                rel_force = np.abs((force - old_force) / old_force)
                logging.info(f'rel. force change: {rel_force:.2e}')
                if (rel_force < F_TOL) or (I_incr >= MAX_STEPS):
                    raise ValueError
                I_incr += I_incr
                F_incr += F_incr

            except ValueError:
                self.sys.force_cap = 0
                logging.info('explosions avoided sucessfully!')
                break

    def magnetize(self, part_list, dip_magnitude, H_ext):
        '''
        Apply the langevin magnetisation law to determine the magnitude of the dipole moment of each particle in part_list, projected along H_tot=H_ext+tot_dip_fld. part_list should be a iterable that contains espresso particleHandle objects.

        :param part_list: iterable(ParticleHandle) | ParticleSlice could work but prefer to wrap with the list() constructor.
        :param dip_magnitude: float
        :param H_ext: float

        :return: None

        '''
        for part in part_list:
            H_tot = part.dip_fld+H_ext
            tri = np.linalg.norm(H_tot)
            dip_tri = dip_magnitude*tri
            inv_dip_tri = 1.0/(dip_tri)
            inv_tanh_dip_tri = 1.0/np.tanh(dip_tri)
            part.dip = dip_magnitude/tri*(inv_tanh_dip_tri-inv_dip_tri)*H_tot
            logging.info(part.dip)

    def set_H_ext(self, H=(0, 0, 1.)):
        """
        Sets an espressomd.constraints.HomogeneousMagneticField in the simulation. Will delete any other HomogeneousMagneticField constraint if present. Safe to use for rotating or AC magnetic fileds.

        :param H: tuple | The external magnetic field vector. Default is (0, 0, 1).
        :return: None
        """
        for x in self.sys.constraints:
            if isinstance(x,espressomd.constraints.HomogeneousMagneticField):
                logging.info(f'Removed old H: {x}')
                self.sys.constraints.remove(x)
        ExtH = espressomd.constraints.HomogeneousMagneticField(H=list(H))
        self.sys.constraints.add(ExtH)
        logging.info(f'External field set: {ExtH.H}')

    def get_H_ext(self):
        """
        Retrieves the current external magnetic field. Assumes there is only one active applied field!

        :return: tuple | The external magnetic field vector.
        """
        HFld=next(x for x in self.sys.constraints if isinstance(x,espressomd.constraints.HomogeneousMagneticField))
        return HFld.H

    def init_pickle_dump(self, path_to_dump):
        """
        Initializes a pickle dump file to store simulation data.

        This method creates a new compressed pickle file at the specified path and initializes it with an empty dictionary.

        :param path_to_dump: str | Path where the pickle dump file should be created.
        :return: tuple | A tuple containing the path to the dump file and an initial counter value (0).
        """
        dict_of_god = {}
        f = gzip.open(path_to_dump, 'wb')
        pickle.dump(dict_of_god, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return path_to_dump, 0

    def load_pickle_dump(self, path_to_dump):
        """
        Loads simulation data from a pickle dump file.

        Reads a compressed pickle file from the specified path and returns the data along with the next timestep counter.

        :param path_to_dump: str | Path to the pickle dump file.
        :return: tuple | A tuple containing the path to the dump file and the next timestep counter based on the loaded data.
        """
        f = gzip.open(path_to_dump, 'rb')
        dict_of_god = pickle.load(f)
        f.close()
        return path_to_dump, int(list(dict_of_god.keys())[-1].split('_')[-1])+1

    def dump_to_init(self, path_to_dump, dungeon_witch_list, cnt):
        """
        Appends simulation data for a given timestep to an existing pickle dump.

        This method reads data from a compressed pickle file, adds new data for the specified timestep, and writes the updated data back to the file.

        :param path_to_dump: str | Path to the pickle dump file.
        :param dungeon_witch_list: list | A list of objects to be serialized and stored for the current timestep.
        :param cnt: int | The current timestep counter to be used as a key in the dump.
        :return: None
        """
        f = gzip.open(path_to_dump, 'rb')
        dict_of_god = pickle.load(f)
        f.close()
        dict_of_god['timestep_%s' % cnt] = [x.to_dict_of_god()
                                            for x in dungeon_witch_list]
        f = gzip.open(path_to_dump, 'wb')
        pickle.dump(dict_of_god, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def generate_positions(self, min_distance):
        """
        Generates random positions for objects in the simulation box, ensuring minimum distance between positions. Completely naive implementation

        :param min_distance: float | The minimum allowed distance between objects.
        :return: np.ndarray | Array of generated positions.
        """
        object_positions = []
        while len(object_positions) < self.no_objects:
            new_position = np.random.random(3) * self.sys.box_l
            if all(np.linalg.norm(new_position - pos) >= min_distance for pos in self.sys.part.all().pos):
                if all(np.linalg.norm(new_position - existing_position) >= min_distance for existing_position in object_positions):
                    object_positions.append(new_position)
            logging.info(f'position casing progress: {len(object_positions)/self.no_objects}')

        return np.array(object_positions)

import espressomd
from espressomd import shapes
import espressomd.version
if espressomd.version.major() == 4:
    from espressomd.virtual_sites import VirtualSitesRelative
from pressomancy.analysis import H5DataSelector
import sys as sysos
import numpy as np
import os
import gzip
import pickle
from itertools import combinations_with_replacement
from pressomancy.object_classes import *
from pressomancy.helper_functions import *
import logging
import h5py
from collections import Counter
import shutil
from pathlib import Path
import inspect

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

    Notes:
        - The class assumes that the ESPResSo system is already instantiated and wraps the system handle during initialization. The initialisation and lifetime is managed by the decorator class.
        - Many methods rely on specific attributes or methods being implemented in the stored objects. This is why any object that is to be safely used by Simulation should use the SimulationObject metaclass.
        - This class is designed to be extensible for different types of interactions and constraints.
    """
    _allowed_direct_set = {"_allowed_direct_set", "seed", "kT"}

    _object_permissions=['part_types']
    _sys=espressomd.System
    def __init__(self, box_dim, use_espresso_checkpoint_system=None):
        # Private attributes (# with public properties)
        self._no_objects = 0 # .no_objects
        self._objects = [] # .objects
        self._part_types = PartDictSafe({}) # .part_types
        self._partitioned=None
        self._part_positions=[]
        self._volume_size=None
        self._volume_centers=[]

        # Public attributes
        # espresso system is accessed by .sys, e.g. self.sys.part.all()
        # I/O
        self.io_dict={'h5_file': None,'properties':[('id',1), ('type',1), ('pos',3),('pos_folded',3), ('director',3),('image_box',3), ('f',3),('dip',3)], 'bonds':None,'flat_part_view':defaultdict(list),'registered_group_type': None}
        self.src_params_set=False

        # System numbers stuff
        self.seed = int.from_bytes(os.urandom(2), sysos.byteorder)
        self.kT = 1.

    @property
    def no_objects(self):
        return self._no_objects
    @property
    def objects(self):
        return self._objects
    @property
    def part_types(self):
        return self._part_types

    def __setattr__(self, name, value):
        # Allow initialization
        if not hasattr(self, name):
            object.__setattr__(self, name, value)
            return
        
        # allow internal properties (start with "_")
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
            
        # Block protected and important attributes
        protected_attrs = {
            'no_objects': "Object count is managed automatically",
            'part_types': "Dictionairy that stores particle type names and type integers is managed automatically. Only after a checkpoint should you use system.part_types.update() to restore checkpointed part_types dict.",
            'objects': "Object list is managed automatically. Use store_objects() or other Simulation method to store objects.",
            'sys': "System is read-only. Use rebind_sys() - only should be used after loading an espresso checkpoint."
        }
        
        if name in self._allowed_direct_set:
            object.__setattr__(self, name, value)
        elif name in protected_attrs:
            raise AttributeError(protected_attrs[name])
        else:
            raise AttributeError(
                f"Cannot set '{name}' directly. Use appropriate methods."
            )    
    
    def set_init_src(self, path, pos_ori_src_type=['real',], type_to_type_map=[], prop_to_prop_map=[], declare_types=[]):
        self.src_path_h5=path
        self.pos_ori_src_type=pos_ori_src_type
        self.type_to_type_map=type_to_type_map
        self.prop_to_prop_map=prop_to_prop_map
        self.src_params_set=True
        for typ_decl in declare_types:
            for x,y in typ_decl.items():
                self._part_types[x]=y
    
    def set_sys(self, time_step=0.01, min_global_cut=3.0, have_quaternion=False):
        '''
        Set espresso cellsystem params, and import virtual particle scheme. Run automatically on initialisation of the System class.
        '''
        np.random.seed(seed=self.seed)
        logging.info(f'core.seed: {self.seed}')
        self.sys.periodicity = (True, True, True)
        self.sys.time_step = time_step
        self.sys.cell_system.skin = 0.5
        self.sys.min_global_cut = min_global_cut
        if espressomd.version.major()==4:
            self.sys.virtual_sites = VirtualSitesRelative(have_quaternion=have_quaternion)
        assert self.api_agnostic_feature_check('VIRTUAL_SITES_RELATIVE'), 'VirtualSitesRelative must be set. If not, anything involving virtual particles will not work correctly, but it might be very hard to figure out why. I have wasted days debugging issues only to remember i commented out this line!!!'
        logging.info(f'System params have been autoset. The values of min_global_cut and skin are not guaranteed to be optimal for your simualtion and should be tuned by hand!!!')

    def modify_system_attribute(self, requester, attribute_name, action):
        """
        Validates and modifies a Simulation attribute if allowed by the permissions.

        :param requester: The object requesting the modification.
        :param attribute_name: str | The name of the attribute to modify.
        :param action: callable | A function that takes the current attribute value as input and modifies it.
        :return: None
        """
        if hasattr(self, attribute_name) and attribute_name in self._object_permissions:
            action(getattr(self,attribute_name))

        else:
            logging.info("Requester does not have permission to modify attributes.")

    def reset_non_bonded_inter(self):
        """
        Resets wca interactions (uncomment to add more). Removes only interactions between types from pressomancy objects.
        
        Workaround until espressomd.BondedInteractions.reset() is fixed.
        """
        for (type1, type2) in combinations_with_replacement(tuple(self._part_types.values()), 2):
            self.sys.non_bonded_inter[type1,type2].wca.deactivate()

            # self.sys.non_bonded_inter[type1,type2].tabulated.deactivate()
            # self.sys.non_bonded_inter[type1,type2].lennard_jones.deactivate()
            # self.sys.non_bonded_inter[type1,type2].generic_lennard_jones.deactivate()
            # self.sys.non_bonded_inter[type1,type2].lennard_jones_cos.deactivate()
            # self.sys.non_bonded_inter[type1,type2].lennard_jones_cos2.deactivate()
            # self.sys.non_bonded_inter[type1,type2].smooth_step.deactivate()
            # self.sys.non_bonded_inter[type1,type2].bmhtf.deactivate()
            # self.sys.non_bonded_inter[type1,type2].morse.deactivate()
            # self.sys.non_bonded_inter[type1,type2].buckingham.deactivate()
            # self.sys.non_bonded_inter[type1,type2].soft_sphere.deactivate()
            # self.sys.non_bonded_inter[type1,type2].hat.deactivate()
            # self.sys.non_bonded_inter[type1,type2].hertzian.deactivate()
            # self.sys.non_bonded_inter[type1,type2].gaussian.deactivate()
            # self.sys.non_bonded_inter[type1,type2].dpd.deactivate()
            # self.sys.non_bonded_inter[type1,type2].thole.deactivate()
            # self.sys.non_bonded_inter[type1,type2].gay_berne.deactivate()

    def sanity_check(self,object):
        '''
        Method that checks if the object has the required features to be stored in the simulation. If the object has the required features it is stored in the self.objects list.
        '''
        missing_features = set(object.required_features) - set(espressomd.features())
        if missing_features:
            raise MissingFeature(f"Missing required features: {', '.join(missing_features)}")

    def store_objects(self, iterable_list, report=True):
        '''
        Method stores objects in the self.objects dict, if the object has a n_part and part_types attributes,
        and the list of objects passed to the method is commesurate with the system level attribute n_tot_parts.
        Populates the self.part_types attribute with types found in the objects that are stored.
        All objects that are stored should have the same types stored, but this is not checked explicitly
        '''
        temp_dict={}
        for element in iterable_list:
            if element.params['associated_objects'] != None:
                check_any=any(associated in self._objects for associated in element.params['associated_objects'])
                if check_any:
                    check_all=all(associated in self._objects for associated in element.params['associated_objects'])
                    if not check_all:
                        raise ValueError(f"Some associated objects {element.params['associated_objects']} but not all  associated objects are stored in the simulation. This is a sign that smth major is fucked...Suffer in silence.")
                else:
                    self.store_objects(element.params['associated_objects'],report=False)
            assert element not in self._objects, "Lists have common elements!"
            self.sanity_check(element)
            element.modify_system_attribute = self.modify_system_attribute
            self._objects.append(element)
            for key, val in element.part_types.items():
                temp_dict[key]=val
            self._no_objects += 1
        self._part_types.update(temp_dict)
        if report:
            names = [element.__class__.__name__ for element in self._objects]
            counts = Counter(names)
            formatted = ", ".join(f"{count} {name}" for name, count in counts.items())
            logging.info(f"{formatted} stored")

    def set_objects(self, objects, box_lengths=None, shift=[0,0,0], mode='NEW'):
        """Set objects' positions and orientations in a box. Defaults to the Simulation box.
        This method places objects in the simulation box using a partitioning scheme. For the first placement, it generates exactly the required number of positions. For subsequent placements, it searches for non-overlapping positions with existing objects. This guarantees non-overlapping of the objects.
        Parameters
        ----------
        objects : list
            A list of simulation objects to place. All objects must be instances of the same type.
        box_lengths : array-like of shape (3,), optional
            Dimensions of the box into which the objects will be placed. If not provided,
            the default system box dimensions (`self.sys.box_l`) are used.
        shift : array-like of shape (3,), optional
            A vector by which to shift all placed object positions. Default is [0, 0, 0].

        Raises
        ------
        AssertionError
            If not all objects are of the same type.
        NotImplementedError
            If trying to place objects when more than one previous partition exists.
        Notes
        -----
        The current implementation supports placing objects either in an empty system or in a system with exactly one previous partition. The method uses partition_cuboid_volume to generate positions and orientations, and for subsequent placements, ensures no overlaps with existing objects through get_cross_lattice_nonintersecting_volumes. The method automatically adjusts the search space (by increasing the factor) if it cannot find enough non-overlapping positions in subsequent placements.
        """
        if box_lengths is None:
            box_lengths = self.sys.box_l
        box_lengths = np.asarray(box_lengths)
        shift = np.asarray(shift)
        
        # Ensure all objects are of the same type.
        assert all(isinstance(item, type(objects[0])) for item in objects), "Not all items have the same type!"
        if mode=="INIT_SRC":
            positions, orientations=self.get_pos_ori_from_src(objects)
        else:
            # centeres, polymer_positions = partition_cuboid_volume_oriented_rectangles(big_box_dim=self.sys.box_l, num_spheres=len(filaments), small_box_dim=np.array([filaments[0].sigma, filaments[0].sigma, filaments[0].size]), num_monomers=filaments[0].n_parts)
            if len(self._part_positions)== 0:
                # First placement: generate exactly len(objects) positions.
                centeres, positions, orientations = partition_cuboid_volume(
                    box_lengths=box_lengths,
                    num_spheres=len(objects),
                    sphere_diameter=objects[0].params['size'],
                    routine_per_volume=objects[0].build_function
                )
                self._volume_centers.append(centeres)
                self._part_positions.append(positions)
                self._volume_size = objects[0].params['size']
            elif len(self._part_positions) == 1:
                # Subsequent placements: search for positions without overlaps.
                if not all(box_lengths[0]==llen for llen in box_lengths) and box_lengths[0]==self.sys.box_l[0]:
                    raise NotImplemented("Currently box_lenghts must be equal to system box size, for consecutive usages of set_objects.")
                factor = 1
                while True:
                    centeres, positions, orientations = partition_cuboid_volume(
                        box_lengths=box_lengths,
                        num_spheres=len(objects) * factor,
                        sphere_diameter=objects[0].params['size'],
                        routine_per_volume=objects[0].build_function
                    )
                    res=get_cross_lattice_nonintersecting_volumes(
                        current_lattice_centers=centeres,
                        current_lattice_grouped_part_pos=positions,
                        current_lattice_diam=objects[0].params['size'],
                        other_lattice_centers=self._volume_centers[0],
                        other_lattice_grouped_part_pos=self._part_positions[0],
                        other_lattice_diam=self._volume_size,
                        box_lengths=box_lengths
                        )
                    mask=[key for key,val in res.items() if all(val)]
                    positions=positions[mask]
                    orientations=orientations[mask]
                    if len(positions) >= len(objects):
                        break
                    else :
                        factor += 1
                        logging.info('Failed to find enough space; (found, needed): (%d, %d). Will retry by requesting %d times the number of parts', len(positions), len(objects),factor)
            else:
                raise NotImplementedError('The repartitioning scheme can currently handle only the case where one previos partition exists. More than than is still not supported')
        
        positions += shift
        for obj, pos, ori in zip(objects, positions, orientations):
            obj.set_object(pos, ori)
        names = [element.__class__.__name__ for element in objects]
        counts = Counter(names)
        formatted = ", ".join(f"{count} {name}" for name, count in counts.items())
        logging.info(f"{formatted} set!!!")

    def place_objects(self, objects, positions, orientations=None):
        """Set objects' positions and orientations in a box.
        This method places objects at given coordinates within the simulation box and sets their orientations.
        If orientations are not provided, random unit vectors are generated.
        This method does not guarantee non-overlapping of objects, in any way.

        Parameters
        ----------
        objects : list or array-like
            List of simulation objects to place. Can be a single object or multiple.
        positions : array-like of shape (N, 3)
            A list or array of 3D coordinates where each object will be placed.
        orientations : array-like of shape (N, 3), optional
            Orientation vectors for each object. If not provided, random unit vectors are generated.

        Raises
        ------
        AssertionError
            If the number of objects, positions, and orientations do not match.
        """
        objects= np.array([objects]).ravel()
        if orientations is None:
            orientations = generate_random_unit_vectors(len(positions))
        else:
            orientations = normalize_vectors(orientations)
        assert len(objects) == len(positions) == len(orientations)
        for obj, pos, ori in zip(objects, positions, orientations):
            obj.set_object(pos, ori)
        logging.info('%s placed!!!', objects[0].__class__.__name__)

    def set_objects_god(self, objects, positions, orientations=None, **kwargs):
        """Set objects' everything in the simulation box.
        This method places objects at given coordinates within the simulation box and sets their orientations. Furthermore, it sets any other object/particle properti passed through as extra keyword arguments.
        This method does not guarantee non-overlapping of objects, in any way.

        Parameters
        ----------
        objects : list or array-like
            List of simulation objects to place. Can be a single object or multiple.
        positions : array-like of shape (N, 3)
            A list or array of 3D coordinates where each object will be placed.
        orientations : array-like of shape (N, 3)
            Orientation vectors for each object
        **kwargs : keyword arguments (any number)
            Valid object.set_object() or espressomd.part.add() keyword arguments.

        Raises
        ------
        AssertionError
            If the number of objects, positions, orientations, and every kwargs item lenght do not match.
        """
        objects= np.array([objects]).ravel()
        positions= np.atleast_2d(positions)
        len_objects=len(objects)

        if orientations is None:
            orientations = np.zeros_like(positions) + [0,0,1]
        else:
            orientations = normalize_vectors(orientations)
        orientations= np.atleast_2d(orientations)
        assert len_objects == len(positions) == len(orientations)
        for key in kwargs.keys():
            kwargs[key] = broadcast_to_len(len_objects, kwargs[key])
        kwargs_keys = kwargs.keys()
        for obj, pos, ori, *kwa_values in zip(objects, positions, orientations, *kwargs.values()):
            kwa = dict(zip(kwargs_keys, kwa_values))
            obj.set_object(pos, ori, **kwa)
        logging.info('%s placed!!!', objects[0].__class__.__name__)

    def place_objects(self, objects, positions, orientations=None):
        """Set objects' positions and orientations in a box.
        This method places objects at given coordinates within the simulation box and sets their orientations.
        If orientations are not provided, random unit vectors are generated.
        This method does not guarantee non-overlapping of objects, in any way.

        Parameters
        ----------
        objects : list or array-like
            List of simulation objects to place. Can be a single object or multiple.
        positions : array-like of shape (N, 3)
            A list or array of 3D coordinates where each object will be placed.
        orientations : array-like of shape (N, 3), optional
            Orientation vectors for each object. If not provided, random unit vectors are generated.

        Raises
        ------
        AssertionError
            If the number of objects, positions, and orientations do not match.
        """
        objects= np.array([objects]).ravel()
        if orientations is None:
            orientations = generate_random_unit_vectors(len(positions))
        else:
            orientations = normalize_vectors(orientations)
        assert len(objects) == len(positions) == len(orientations)
        for obj, pos, ori in zip(objects, positions, orientations):
            obj.set_object(pos, ori)
        logging.info('%s placed!!!', objects[0].__class__.__name__)

    def set_objects_god(self, objects, positions, orientations=None, **kwargs):
        """Set objects' everything in the simulation box.
        This method places objects at given coordinates within the simulation box and sets their orientations. Furthermore, it sets any other object/particle properti passed through as extra keyword arguments.
        This method does not guarantee non-overlapping of objects, in any way.

        Parameters
        ----------
        objects : list or array-like
            List of simulation objects to place. Can be a single object or multiple.
        positions : array-like of shape (N, 3)
            A list or array of 3D coordinates where each object will be placed.
        orientations : array-like of shape (N, 3)
            Orientation vectors for each object
        **kwargs : keyword arguments (any number)
            Valid object.set_object() or espressomd.part.add() keyword arguments.

        Raises
        ------
        AssertionError
            If the number of objects, positions, orientations, and every kwargs item lenght do not match.
        """
        objects= np.array([objects]).ravel()
        positions= np.atleast_2d(positions)
        len_objects=len(objects)

        if orientations is None:
            orientations = np.zeros_like(positions) + [0,0,1]
        else:
            orientations = normalize_vectors(orientations)
        orientations= np.atleast_2d(orientations)
        assert len_objects == len(positions) == len(orientations)
        for key in kwargs.keys():
            kwargs[key] = broadcast_to_len(len_objects, kwargs[key])
        kwargs_keys = kwargs.keys()
        for obj, pos, ori, *kwa_values in zip(objects, positions, orientations, *kwargs.values()):
            kwa = dict(zip(kwargs_keys, kwa_values))
            obj.set_object(pos, ori, **kwa)
        logging.info('%s placed!!!', objects[0].__class__.__name__)

    def mark_for_collision_detection(self, object_type=Quadriplex, part_type=666):
        assert any(isinstance(ele, object_type) for ele in self._objects), "method assumes simulation holds correct type object"

        self._part_types['marked'] = 666
        objects_iter = [ele for ele in self._objects if isinstance(ele, object_type)]
        assert all((hasattr(ob, 'mark_covalent_bonds') and callable(getattr(ob, 'mark_covalent_bonds')))
                   for ob in objects_iter), "method requires that stored objects have mark_covalent_bonds() method"
        for obj_el in objects_iter:
            obj_el.mark_covalent_bonds(part_type=part_type)

    def init_magnetic_inter(self, actor_handle):

        if espressomd.version.major()==4:
            self.sys.actors.clear()
            self.sys.actors.add(actor_handle)
        elif espressomd.version.major()==5:
            self.sys.magnetostatics.clear()
            self.sys.magnetostatics.solver = actor_handle
        else:
            raise NotImplementedError('Only ESPResSo 4 and 5 are supported')

        logging.info(f'{actor_handle} magnetic interactions actor initiated')

    def set_steric(self, key=('nonmagn',), wca_eps=1., sigma=1.):
        '''
        Set WCA interactions between particles of types given in the key parameter.
        :param key: tuple of keys from self.part_types | Default only nonmagn WCA
        :param wca_epsilon: float | strength of the steric repulsion.

        :return: None

        Interaction length is allways determined from sigma.
        '''
        logging.info(f'part types available {self._part_types.keys()} ')
        logging.info(f'WCA interactions initiated for keys: {key}')
        for key_el, key_el2 in combinations_with_replacement(key, 2):
            self.sys.non_bonded_inter[self._part_types[key_el], self._part_types[key_el2]
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
            self.sys.non_bonded_inter[self._part_types[key_el], self._part_types[key_el2]
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
            self.sys.non_bonded_inter[self._part_types[key_el], self._part_types[key_el2]].lennard_jones.set_params(
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
            self.sys.non_bonded_inter[self._part_types[key_el], self._part_types[key_el2]].lennard_jones.set_params(
                epsilon=eps, sigma=sgm, cutoff=lj_cut, shift=0)
        logging.info('vdW interactions initiated!')

    def add_box_constraints(self, wall_type=0, sides=['all'], inter=None, types_=None, object_types=None,
                        bottom=None, top=None, left=None, right=None, back=None, front=None):
        """
        Adds wall constraints to the simulation box along specified sides.

        This method calls helper_functions.add_box_constarints to place flat wall constraints (using `espressomd.shapes.Wall`) perpendicular to the box axes, typically used to confine particles within the simulation domain. By default, walls are added on all six faces of the box. You can customize which walls to include or exclude, their positions, and interaction types with other particles.
        By default:
            bottom - z=0; top - z=self.sys.box_l[2];
            left - y=0  ; right - y=self.sys.box_l[1];
            back - x=0  ; front - z=self.sys.box_l[0];

        Parameters
        ----------
        wall_type : int, optional
            Particle type used for the wall (default: 0).
        sides : list of str, optional
            Specifies which sides to add walls on. Default is ['all'], which includes all six box faces.
            Supported values:
                - 'all': add walls on all six faces.
                - 'sides': add walls on all but the top and bottom.
                - Individual sides: 'top', 'bottom', 'left', 'right', 'front', 'back'.
                - 'no-<side>': exclude specific sides, e.g., 'no-top', 'no-right', 'no-sides'.
        inter : str or list of str, optional
            Type(s) of interaction to enable between wall and specified particle types. Currently supports:
                - 'wca': Weeks–Chandler–Andersen potential with large epsilon.
        types_ : list of int, optional
            Particle types that will interact with the walls. If None, all non-wall types in the system are used.
        bottom, top, left, right, back, front : float, optional
            Position of each wall, defined as the distance to the xOy plane (for top/bottom), xOz plane (for left/right),
            or yOz plane (for front/back). If not specified, the position defaults to the corresponding boundary of the simulation box.


        Returns
        -------
        list of espressomd.constraints.ShapeBasedConstraint
            List of wall constraint objects added to the system. (can be used to later specify which walls to remove).
            Organized as: bottom->top->left->right->back->front

        Notes
        -----
        - If `sides` includes any entry starting with 'no-', that side will be excluded even if 'all' or 'sides' is specified.
        - The wall interaction can be configured by specifying `inter` and, optionally, `types_`.
        - Walls are defined using outward-pointing normals and placed at specified distances from the origin.
        - The method adds constraints to `self.sys.constraints` directly.
        """
        wall_constraints = add_box_constraints_func(self.sys, wall_type=wall_type, sides=sides, inter=inter, types_=types_, object_types=object_types, bottom=bottom, top=top, left=left, right=right, back=back, front=front)

        return wall_constraints
    
    def remove_box_constraints(self, wall_constraints=None, part_types=None, object_types=None):
        """ Removes wall_constraints from system. Default: removes all espressomd.shapes.Wall constraints.
            If part_types is not None, remove only interactions with those particle types.

            Calls helper_functions.remove_box_contraints
        system
        list of espressomd.constraints.ShapeBasedConstraint wall_constraints
        list of particles types to stop interactoin with box part_types
        """
        remove_box_constraints_func(self.sys, wall_constraints=wall_constraints, part_types=part_types, object_types=object_types)
        

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
        if not self.api_agnostic_feature_check('WALBERLA'):
            name = f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}"
            raise MissingFeature(f"{name} requires WALBERLA. Please enable it in your ESPResSo installation.")
        self.sys.thermostat.turn_off()
        self.sys.part.all().v = (0, 0, 0)
        param_dict={'kT':kT, 'seed':self.seed, 'agrid':agrid, 'dens':dens, 'visc':visc, 'tau':timestep}
        if self.api_agnostic_feature_check('CUDA'):
            logging.info('GPU LB method is beeing initiated')

            lbf = espressomd.lb.LBFluidWalberlaGPU(**param_dict)
        else:
            logging.info('CPU LB method is beeing initiated')
            lbf = espressomd.lb.LBFluidWalberla(**param_dict)
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
        if not self.api_agnostic_feature_check('LB_BOUNDARIES'):
            name = f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}"
            raise MissingFeature(f"{name} requires LB_BOUNDARIES. Please enable it in your ESPResSo installation.")

        logging.info("Setup LB boundaries.")
        top_wall = shapes.Wall(normal=[1, 0, 0], dist=1) # type: ignore
        bottom_wall = shapes.Wall( # type: ignore
            normal=[-1, 0, 0], dist=-(self.sys.box_l[0] - 1))

        top_boundary = espressomd.lbboundaries.LBBoundary( # type: ignore
            shape=top_wall, velocity=slip_vel)
        bottom_boundary = espressomd.lbboundaries.LBBoundary(shape=bottom_wall) # type: ignore

        self.sys.lbboundaries.add(top_boundary)
        self.sys.lbboundaries.add(bottom_boundary)

    def avoid_explosion(self, F_TOL, MAX_STEPS=5, F_incr=100, I_incr=100):
        """
        Iteratively caps forces to prevent simulation instabilities.
        :param F_TOL: float | Force change tolerance between iterations to determine convergence.
        :param MAX_STEPS: int | Maximum number of steps for force iteration. Default is 5.
        :param F_incr: int | Amount to increase force cap by each iteration. Default is 100.
        :param I_incr: int | Amount to increase integration steps by each iteration. Default is 100.
        :return: None

        The method gradually increases both the force cap and integration timestep while monitoring the relative force change between iterations. If the relative change falls below F_TOL or MAX_STEPS is reached, the iteration stops.
        """
        timestep_og=self.sys.time_step
        timestep_icr=timestep_og/MAX_STEPS
        logging.info('iterating with a force cap.')
        self.sys.integrator.run(0)
        STEP=1
        while True:
            self.sys.time_step=timestep_icr*STEP
            old_force = np.max(np.linalg.norm(
                self.sys.part.all().f, axis=1))
            self.sys.force_cap = F_incr
            self.sys.integrator.run(I_incr)
            force = np.max(np.linalg.norm(self.sys.part.all().f, axis=1))
            rel_force = np.abs((force - old_force) / old_force)
            logging.info(f'rel. force change: {rel_force:.2e}')
            if (rel_force < F_TOL) or (STEP >= MAX_STEPS):
                break
            STEP += 1
            I_incr += I_incr
            F_incr += F_incr

        self.sys.force_cap = 0
        self.sys.time_step=timestep_og
        logging.info('explosions avoided sucessfully!')

    def magnetize(self, part_list, dip_magnitude, H_ext):
        '''
        Apply the langevin magnetisation law to determine the magnitude of the dipole moment of each particle in part_list, projected along H_tot=H_ext+tot_dip_fld. part_list should be a iterable that contains espresso particleHandle objects.

        :param part_list: iterable(ParticleHandle) | ParticleSlice could work but prefer to wrap with the list() constructor.
        :param dip_magnitude: float
        :param H_ext: float

        :return: None

        '''
        if not self.api_agnostic_feature_check('DIPOLE_FIELD_TRACKING'):
            name = f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}"
            raise MissingFeature(f"{name} requires DIPOLE_FIELD_TRACKING. Please enable it in your ESPResSo installation.")
        for part in part_list:
            H_tot = part.dip_fld+H_ext
            tri = np.linalg.norm(H_tot)
            if tri < 1e-5:
                part.dip = H_tot/tri * 1e-6
            else:
                dip_tri = dip_magnitude*tri #/ self.kT
                inv_dip_tri = 1.0/(dip_tri)
                inv_tanh_dip_tri = 1.0/np.tanh(dip_tri)
                part.dip = dip_magnitude/tri*(inv_tanh_dip_tri-inv_dip_tri)*H_tot
            logging.info(part.dip)

    def magnetize_lin(self, part_list, dip_magnitude, H_ext):
        '''
        Apply a linear magnetisation law to determine the magnitude of the dipole moment of each particle in part_list, projected along H_tot=H_ext+tot_dip_fld. part_list should be a iterable that contains espresso particleHandle objects.

        :param part_list: iterable(ParticleHandle) | ParticleSlice could work but prefer to wrap with the list() constructor.
        :param dip_magnitude: float
        :param H_ext: float

        :return: None

        '''
        assert H_tot < 1., "for magnetic fields above 1, the particle's dipm would surpass their saturation value"
        for part in part_list:
            H_tot = part.dip_fld+H_ext
            tri = np.linalg.norm(H_tot)
            if tri < 1e-5:
                part.dip = H_tot/tri * 1e-6
            else:
                part.dip = dip_magnitude*H_tot
            logging.info(part.dip)

    def magnetize_froelich_kennelly(self, part_list, dip_magnitude, H_ext, Xi=0.5):
        '''
        Apply the empirical Frölich-Kennelly magnetisation law to determine the magnitude of the dipole moment of each particle in part_list, projected along H_tot=H_ext+tot_dip_fld. part_list should be a iterable that contains espresso particleHandle objects.

        :param part_list: iterable(ParticleHandle) | ParticleSlice could work but prefer to wrap with the list() constructor.
        :param dip_magnitude: float
        :param H_ext: float
        :param Xi: float (=0.5) | Susceptibility

        :return: None

        '''
        for part in part_list:
            H_tot = part.dip_fld+H_ext
            tri = np.linalg.norm(H_tot)
            if tri < 1e-5:
                part.dip = H_tot/tri * 1e-6
            else:
                part.dip = Xi*dip_magnitude / (dip_magnitude + Xi*tri) * H_tot
            logging.info(part.dip)

    def magnetize_dumb(self, part_list, dip_magnitude, H_ext):
        '''
        Apply the langevin magnetisation law to determine the magnitude of the dipole moment of each particle in part_list, projected along H_ext. part_list should be a iterable that contains espresso particleHandle objects.

        :param part_list: iterable(ParticleHandle) | ParticleSlice could work but prefer to wrap with the list() constructor.
        :param dip_magnitude: float
        :param H_ext: float

        :return: None

        note: This function does not take into account dipolar fields.

        '''
        for part in part_list:
            tri = np.linalg.norm(H_ext)
            dip_tri = dip_magnitude*tri #/ self.kT
            inv_dip_tri = 1.0/(dip_tri)
            inv_tanh_dip_tri = 1.0/np.tanh(dip_tri)
            part.dip = dip_magnitude/tri*(inv_tanh_dip_tri-inv_dip_tri)*H_ext
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
        Retrieves the current external magnetic field.
        
        Sums over all applied homogeneus magnetic fields.

        :return: tuple | The external magnetic field vector.
        """
        HFld=np.asarray(
            [ele.H for ele in list(self.sys.constraints)
             if isinstance(ele, espressomd.constraints.HomogeneousMagneticField)]
                        ).sum(axis=0)
        return HFld

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
    
    def collect_instances_recursively(self, roots):
        """
        Traverse each root in `roots` and return a flat preorder list
        of every object reachable via `.associated_objects`.
        Raises RuntimeError on any duplicate.
        """
        seen = set()
        result = []

        def traverse(obj):
            if obj in seen:
                raise RuntimeError(f"Duplicate object detected during recursion: {obj!r}")
            seen.add(obj)
            result.append(obj)
            for child in getattr(obj, "associated_objects", []) or []:
                traverse(child)

        for root in roots:
            traverse(root)

        return result

    def inscribe_part_group_to_h5(self, group_type=None, h5_data_path=None,mode='NEW',force_resize_to_size=None):
        """
        Inscribe one or more groups of simulation objects into an HDF5 file.

        This method creates (or opens) an HDF5 file and, for each `group_type`:
        - Builds a flat list of particle handles and their coordinating indices
        - Creates `/particles/<GroupName>` and corresponding property datasets
        - Creates `/connectivity/<GroupName>/ParticleHandle_to_<OwnerClass>` tables
        - Creates `/connectivity/<GroupName>/<Left>_to_<Right>` object–object tables

        Parameters
        ----------
        group_type : list of type
            A list of `SimulationObject` subclasses. All instances of each
            class in `self.objects` will be registered and inscribed.
        h5_data_path : str
            Path to the HDF5 file to write (mode='NEW') or append (mode='LOAD').
        mode : {'NEW', 'LOAD'}, optional
            - 'NEW' : create a fresh file structure (default).
            - 'LOAD': open existing file and resume writing.  

        Returns
        -------
        int
            The starting global counter for writing time steps. Always 0 in
            'NEW' mode; in 'LOAD' mode, the current number of already‑saved steps.

        Raises
        ------
        ValueError
            If `mode` is not one of 'NEW' or 'LOAD'.
        ValueError
            If `group_type` is not a list.
        ValueError
            In 'LOAD' mode, if different groups have mismatched saved step counts.
        """
        if not isinstance(group_type, list):
            raise ValueError("group_type must be a list of classes.")
        if mode not in ('NEW', 'LOAD', 'LOAD_NEW', 'INIT_SRC'):
            raise ValueError(f"Unknown mode: {mode}")
        if force_resize_to_size is not None:
            assert mode=='LOAD_NEW', 'force_resize_to_size can only be used in LOAD_NEW mode'
        self.io_dict['registered_group_type']=[grp_typ.__name__ for grp_typ in group_type]

        if mode in ['NEW', 'INIT_SRC']:
            self.io_dict['h5_file'] = h5py.File(h5_data_path, "w")

            # Create sys group, for espresso system information
            sys_grp = self.io_dict['h5_file'].require_group(f"sys")
            sys_grp.attrs["box_l"] = np.array(self.sys.box_l, dtype=np.float32)
            sys_grp.attrs["periodicity"] = np.array(self.sys.periodicity, dtype=bool)
            sys_grp.attrs["time_step"] = np.float32(self.sys.time_step)


            # Create particles and conecctivity groups for each group type
            par_grp = self.io_dict['h5_file'].require_group(f"particles")
            for grp_typ in group_type:
                data_grp = par_grp.require_group(grp_typ.__name__)
                connect_grp = self.io_dict['h5_file'].require_group(f"connectivity").require_group(grp_typ.__name__)
                logging.info(f"Inscribe: Creating group {grp_typ.__name__} in HDF5 file.")
                objects_to_register=[obj for obj in self._objects if isinstance(obj,grp_typ)]
            
                coordination_indices=[]
                for cr in objects_to_register:
                    part,coord=cr.get_owned_part()
                    self.io_dict['flat_part_view'][grp_typ.__name__].extend(part)
                    coordination_indices.extend(coord)

                total_part_num=len(self.io_dict['flat_part_view'][grp_typ.__name__])

                # Create the connectivity for ParticleHandle to objects that own them.
                grouped = defaultdict(list)
                for part, coords in zip(self.io_dict['flat_part_view'][grp_typ.__name__], coordination_indices):
                    for cls_name, idx in coords:
                        grouped[cls_name].append((part.id, idx))

                for cls_name in sorted(grouped):
                    arr = np.array(grouped[cls_name], dtype=np.int32)
                    connect_grp.create_dataset(
                        f"ParticleHandle_to_{cls_name}",
                        data=arr,
                        dtype=np.int32,
                        maxshape=(arr.shape)
                    )
                # Create the connectivity for objects that own each other
                pair_buckets = defaultdict(list)
                
                for obj in self.collect_instances_recursively(objects_to_register):
                    if not obj.associated_objects:
                        continue
                    left_name = obj.__class__.__name__
                    for sub in obj.associated_objects:
                        right_name = sub.__class__.__name__
                        pair_buckets[(left_name, right_name)].append((obj.who_am_i, sub.who_am_i))

                for (left_name, right_name) in sorted(pair_buckets):
                    arr = np.array(pair_buckets[(left_name, right_name)], dtype=np.int32)
                    ds = connect_grp.create_dataset(
                        f"{left_name}_to_{right_name}",
                        data=arr,
                        dtype=np.int32,
                        maxshape=(arr.shape)
                    )
                # Create the datasets for each property           
                for prop,dim in self.io_dict['properties']:
                    prop_group = data_grp.require_group(prop)
                    prop_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
                    prop_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float32)
                    prop_group.create_dataset(
                        "value",
                        shape=(0, total_part_num, dim),  # Store all particles in a single dataset
                        maxshape=(None, total_part_num, dim),
                        dtype=np.float32,
                        chunks=(1, total_part_num, dim),
                        compression="gzip",
                        compression_opts=4
                    )
                # Create the datasets for bonds (special case, but same structure)
                if self.io_dict.get('bonds') is None:
                    pass
                elif self.io_dict.get('bonds') == "all":
                    # Define bond structure
                    bond_dtype = np.dtype([
                        ("bond_type", h5py.string_dtype(encoding="utf-8")),
                        ("k", np.float32),
                        ("r_0", np.float32),
                        ("r_cut", np.float32),
                        ("partner_id", np.int32),
                    ])
                    vlen_bond_dtype = h5py.vlen_dtype(bond_dtype)

                    prop_group = data_grp.require_group("bonds")
                    prop_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
                    prop_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float32)
                    prop_group.create_dataset(
                        "value",
                        shape=(0, total_part_num),            # same pattern: timestep × particle
                        maxshape=(None, total_part_num),
                        dtype=vlen_bond_dtype,
                        chunks=(1, total_part_num),
                        compression="gzip",
                        compression_opts=4
                    )
                else:
                    raise NotImplemented("Currently only saves no bonds or 'all' bonds.")
                
            GLOBAL_COUNTER=0

        elif mode=='LOAD_NEW':

            self.io_dict['h5_file'] = h5py.File(h5_data_path, "a")
            particles_group = self.io_dict['h5_file']["particles"]
            candidate_lens=[]
            for grp_typ in group_type:
                data_view=H5DataSelector(self.io_dict['h5_file'], particle_group=grp_typ.__name__)
                ids=data_view.get_connectivity_values(grp_typ.__name__)
                part_ids=[]
                for iid in ids:
                    temp=data_view.select_particles_by_object(object_name=grp_typ.__name__,connectivity_value=iid)
                    part_ids+=temp.timestep[-1].id.flatten().tolist()
                part_ids=[int(x) for x in part_ids]
                self.io_dict['flat_part_view'][grp_typ.__name__].extend(self.sys.part.by_ids(part_ids))
                data_grp = particles_group[grp_typ.__name__]
                dataset_val = data_grp["pos/value"]
                candidate_lens.append(dataset_val.shape[0])
            if len(set(candidate_lens)) != 1:
                raise ValueError(
                    f"Inconsistent step counts across groups: {candidate_lens}"
                )
            GLOBAL_COUNTER=candidate_lens[0]

            if force_resize_to_size is not None:
                if self.io_dict['bonds'] is not None:
                    raise NotImplementedError
                assert type(force_resize_to_size) is int, 'force_resize_to_size must be an integer'
                assert force_resize_to_size<=GLOBAL_COUNTER, 'force_resize_to_size must be smaller than or equal to the current number of timesteps saved in file'
                if force_resize_to_size==GLOBAL_COUNTER:
                    logging.info(f'force_resize_to_size is equal to the current number of timesteps saved in file. No resizing will be done.')
                else:
                    for grp_typ in group_type:
                        data_grp = particles_group[grp_typ.__name__]
                        for prop,_ in self.io_dict['properties']:
                            dataset_val = data_grp[f"{prop}/value"]
                            step_dataset = data_grp[f"{prop}/step"]
                            time_dataset = data_grp[f"{prop}/time"]
                            step_dataset.resize((force_resize_to_size,))
                            time_dataset.resize((force_resize_to_size,))
                            dataset_val.resize((force_resize_to_size, dataset_val.shape[1], dataset_val.shape[2]))
                    self.io_dict['h5_file'].flush()
                    logging.info(f'Force resized all datasets from {GLOBAL_COUNTER} to size {force_resize_to_size}')
                    GLOBAL_COUNTER=force_resize_to_size
            logging.info(f"Loaded h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")
        
        elif mode=='LOAD':
            self.io_dict['h5_file'] = h5py.File(h5_data_path, "a")
            particles_group = self.io_dict['h5_file']["particles"]
            candidate_lens=[]
            for grp_typ in group_type:
                objects_to_register=[obj for obj in self._objects if isinstance(obj,grp_typ)]
                for cr in objects_to_register:
                    part,_=cr.get_owned_part()
                    self.io_dict['flat_part_view'][grp_typ.__name__].extend(part)
                data_grp = particles_group[grp_typ.__name__]
                dataset_val = data_grp["pos/value"]
                candidate_lens.append(dataset_val.shape[0])
            if len(set(candidate_lens)) != 1:
                raise ValueError(
                    f"Inconsistent step counts across groups: {candidate_lens}"
                )
            GLOBAL_COUNTER=candidate_lens[0]
            logging.info(f"Loading h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")

        return GLOBAL_COUNTER
        
    def write_part_group_to_h5(self, time_step=None, unique_time=True, bonds_once=True):
        assert self.io_dict['h5_file']!=None,'storage file has not been inscribed!'
        particles_group = self.io_dict['h5_file']["particles"]
        for grp_typ in self.io_dict['registered_group_type']:
            data_grp = particles_group[grp_typ]
            for prop,_ in self.io_dict['properties']:
                dataset_val = data_grp[f"{prop}/value"]
                step_dataset = data_grp[f"{prop}/step"]
                time_dataset = data_grp[f"{prop}/time"]
                if unique_time and len(step_dataset) > 0 and time_step <= step_dataset[-1]:
                    idx = np.searchsorted(step_dataset[:], time_step)
                else:
                    step_dataset.resize((dataset_val.shape[0] + 1,))
                    time_dataset.resize((dataset_val.shape[0] + 1,))
                    dataset_val.resize((dataset_val.shape[0] + 1, dataset_val.shape[1], dataset_val.shape[2]))
                    idx = -1
                step_dataset[idx] = time_step
                time_dataset[idx] = time_step
                dataset_val[idx, :, :] = np.array([np.atleast_1d(getattr(part, prop)) for part in self.io_dict['flat_part_view'][grp_typ]], dtype=np.float32) # TO IMPLEMENT make this type see the rpious and copy. Change the initial type to match type of saved prop
            
            # skip if not saving bonds or if there are already bonds saved and bond_once is True
            if self.io_dict.get('bonds') is None or (bonds_once and data_grp["bonds/value"].shape[0] > 0):
                pass
            elif self.io_dict['bonds'] == "all":
                dataset_val = data_grp["bonds/value"]
                step_dataset = data_grp["bonds/step"]
                time_dataset = data_grp["bonds/time"]
                if unique_time and len(step_dataset) and time_step <= step_dataset[-1]:
                    idx = np.searchsorted(step_dataset[:], time_step)
                else:
                    step_dataset.resize((dataset_val.shape[0] + 1,))
                    time_dataset.resize((dataset_val.shape[0] + 1,))
                    dataset_val.resize((dataset_val.shape[0] + 1, dataset_val.shape[1]))
                    idx = -1
                step_dataset[idx] = time_step
                time_dataset[idx] = time_step
                bond_dtype = np.dtype([
                    ("bond_type", h5py.string_dtype(encoding="utf-8")),
                    ("k", np.float32),
                    ("r_0", np.float32),
                    ("r_cut", np.float32),
                    ("partner_id", np.int32),
                ])
                for i, part in enumerate(self.io_dict['flat_part_view'][grp_typ]):
                    bond_list = []
                    for bond_obj, partner_id in getattr(part, 'bonds', []):
                        bond_list.append((
                            type(bond_obj).__name__,
                            bond_obj.k,
                            bond_obj.r_0,
                            bond_obj.r_cut,
                            partner_id
                        ))
                    if len(bond_list) > 0:
                        dataset_val[idx, i] = np.asarray(bond_list, dtype=bond_dtype)
            else:
                raise NotImplemented("Currently only saves no bonds or 'all' bonds.")

        logging.info(f"Successfully wrote timestep for {self.io_dict['registered_group_type']}.")

    def mk_src_file(self, original_data_file_path, dest_h5_file_path, prop_dim=None, time_step=-1):
        """
        Copy an HDF5 simulation file, shrink it to a single time step, and optionally add one-frame datasets for new particle properties.

        The operation runs in two phases:

        1) **Copy & shrink to one frame**  
        The file at ``original_data_file_path`` is copied to ``dest_h5_file_path``.
        For every group under ``/particles/<Group>/<Prop>``, the datasets
        ``value``, ``step``, and ``time`` are sliced at ``time_step`` and then
        **resized to length 1** (T=1), preserving the chosen frame as the only
        frame in the destination file.

        2) **Optionally create new properties (single frame)**  
        If ``prop_dim`` is provided, for each group name in
        ``self.io_dict['registered_group_type']`` this function creates a new
        property group ``/particles/<Group>/<prop>`` with the standard layout:
        - ``step`` : int32, shape ``(T,)`` (created empty, then resized to 1)
        - ``time`` : float32, shape ``(T,)`` (created empty, then resized to 1)
        - ``value``: float32, shape ``(T, N, D)`` (gzip, chunked as ``(1, N, D)``)
        
        It then appends **one** frame (T=1), setting both ``step[-1]`` and
        ``time[-1]`` to ``time_step``, and fills ``value[-1, :, :]`` from the
        in-memory list ``self.io_dict['flat_part_view'][<Group>]`` using
        ``getattr(part, prop)`` for each particle.

        Parameters
        ----------
        original_data_file_path : str or os.PathLike
            Path to the source HDF5 file to copy.
        dest_h5_file_path : str or os.PathLike
            Destination path for the copied/modified HDF5 file. Parent directories are created if missing.
        prop_dim : iterable[tuple[str, int]] or None, optional
            Iterable of ``(prop_name, dim)`` pairs describing new properties to add as single-frame datasets. If ``None`` (default), the function only performs the copy-and-shrink phase.
        time_step : int, optional
            Index of the frame to keep during the shrink phase and the value written to both ``step`` (int32) and ``time`` (float32) when adding new properties. Must be a valid index for all existing per-property datasets.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If expected groups/datasets (e.g., ``/particles``) are missing.
        IndexError
            If ``time_step`` is out of range for any ``step``/``time``/``value`` dataset.
        ValueError / RuntimeError
            If dataset creation for new properties fails (e.g., attempting to create a dataset that already exists, or a dtype/shape mismatch).
        AssertionError
            If the resulting destination file is not single-step (``len(selector.timestep) != 1``).

        Notes
        -----
        - **Single-step invariant:** After phase (1), the destination file contains exactly one time step (T=1) for all existing properties. The function asserts this using ``H5DataSelector(...).timestep``.
        - **Particle ordering:** New property values are taken from
        ``self.io_dict['flat_part_view'][<Group>]`` in its current order and
        written as an ``(N, dim)`` slab for the single kept frame. This assumes
        that the in-memory order matches the file's particle order.
        - **Creation semantics:** New property datasets are created with
        ``create_dataset``; if a property group already exists, this code will
        raise. Switch to existence checks (e.g., ``if 'value' in prop_group``) or ``require_dataset`` if you need idempotent behavior.
        - **Compression & chunks:** New ``value`` datasets use ``float32`` with
        chunks ``(1, N, dim)`` and ``gzip`` compression level 4 for consistency.

        Examples
        --------
        Copy a file, keep frame ``time_step=0``, and add ``director``/``image_box``:

        >>> self.add_missing_data(
        ...     original_data_file_path="src.h5",
        ...     dest_h5_file_path="dst_single.h5",
        ...     prop_dim=[("director", 3), ("image_box", 3)],
        ...     time_step=0,
        ... )
        """
        
        dst_path=Path(dest_h5_file_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_data_file_path, dst_path)
        with h5py.File(dst_path, "r+") as f:
            grp_particles = f["particles"]
            for group_name in grp_particles:
                g = grp_particles[group_name]
                for _, prop_grp in g.items():
                    val = prop_grp["value"]

                    slice_data = val[time_step, ...]  # shape (1, N, D...)
                    val.resize((1,) + val.shape[1:])
                    val[0, ...] = slice_data

                    ds = prop_grp["step"]
                    step_val = ds[time_step]
                    ds.resize((1,))
                    ds[0] = step_val

                    ds = prop_grp["time"]
                    time_val = ds[time_step]
                    ds.resize((1,))
                    ds[0] = time_val

        print(f"✔ Shrunk to single timestep at: {dst_path}")
        
        if prop_dim != None:
            with h5py.File(dst_path, "a") as h5_file_handle: 
                for grp_typ in self.io_dict['registered_group_type']:
                    particles_group = h5_file_handle["particles"]
                    data_grp = particles_group[grp_typ]
                    total_part_num=len(self.io_dict['flat_part_view'][grp_typ])
                    for prop,dim in prop_dim:
                        prop_group = data_grp.require_group(prop)
                        step_dataset=prop_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
                        time_dataset=prop_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float32)
                        dataset_val=prop_group.create_dataset(
                            "value",
                            shape=(0, total_part_num, dim),  # Store all particles in a single dataset
                            maxshape=(None, total_part_num, dim),
                            dtype=np.float32,
                            chunks=(1, total_part_num, dim),
                            compression="gzip",
                            compression_opts=4
                        )
                        step_dataset.resize((dataset_val.shape[0] + 1,))
                        time_dataset.resize((dataset_val.shape[0] + 1,))
                        dataset_val.resize((dataset_val.shape[0] + 1, dataset_val.shape[1], dataset_val.shape[2]))
                        step_dataset[-1] = time_step
                        time_dataset[-1] = time_step
                        dataset_val[-1, :, :] = np.array([np.atleast_1d(getattr(part, prop)) for part in self.io_dict['flat_part_view'][grp_typ]], dtype=np.float32)
                        src_data_grp = H5DataSelector(h5_file_handle, particle_group=grp_typ)
                        assert len(src_data_grp.timestep)==1,'dataset is ragged!!!'
                        logging.info(f'appended {prop} to {dst_path}')
            
    def set_prop_from_src(
    self,
    registered_objs=None,
    time_step: int = -1,
):
        """
        Update local particle properties from an HDF5 source file for a given group type. This loads a particle group from an HDF5 file, validates that requested type mappings exist both locally and in the source, then iterates over each group instance (connectivity ID) to copy properties from source particles into the corresponding local particles.

        Parameters
        ----------
        registered_objs : iterable
            Collection of local group instances (e.g., Filament objects) whose
            owned particles will be updated.

        time_step : int, optional
            Index of the source time step to read. ``-1`` selects the last frame.

        Notes
        -----
        * For the `(dip -> director)` mapping, vectors are normalized via
        `np.linalg.norm`. Zero-norm dipoles would raise a warning or yield NaNs
        if present—consider guarding if your data can contain zeros.

        Raises
        ------
        AssertionError
            If a mapped type is not present locally or in the source.
        KeyError
            If lookups via `self.part_types` fail (depends on your implementation).
        """

        # Open the source HDF5 and select the data group matching the requested type.
        assert self.src_params_set==True, 'src_params_set must be set before calling this method'
        with h5py.File(self.src_path_h5, "r") as src_file:
            src_data_grp = H5DataSelector(src_file, particle_group=registered_objs[0].__class__.__name__)

            # Discover the set of numeric type IDs present in the source for this group.
            all_src_types_numeric = np.unique(src_data_grp.type)

            # Validate that each requested (src_type -> local_type) exists both locally and in the source file.
            for src_typ, loc_typ in self.type_to_type_map:
                assert (
                    loc_typ in self._part_types or src_typ in self._part_types
                ), (
                    f"local type {loc_typ} or source type {src_typ} not found in "
                    f"simulation part types {self._part_types}"
                )
                assert (
                    self._part_types[src_typ] in all_src_types_numeric
                ), (
                    f"source type {src_typ} with numeric id {self._part_types[src_typ]} "
                    f"not found in source data part types {all_src_types_numeric}"
                )
            logging.info(f"simulation contains types: {self._part_types}")
            logging.info(
                f"src datafile contains types: {self._part_types.key_for(all_src_types_numeric)}"
            )

            # Iterate over each connectivity group (i.e., each distinct instance of the group).
            for loc_obj in registered_objs:

                # Apply each aligned (type mapping, property mapping) pair.
                for (src_typ, loc_typ), (prop_src, prop_loc) in zip(
                    self.type_to_type_map, self.prop_to_prop_map
                ):
                    logging.info(
                        f"Working on {loc_obj.__class__.__name__}: {loc_obj.who_am_i} type {src_typ}->{loc_typ} prop {prop_src}->{prop_loc}"
                    )

                    # Select source particles at the requested time step that belong to this group instance (connectivity == grp_id), and match the numeric type ID mapped from src_typ.
                    part_slice = src_data_grp.timestep[time_step].select_particles_by_object(
                        object_name=loc_obj.__class__.__name__,
                        connectivity_value=loc_obj.who_am_i,
                        predicate=lambda subset: subset.type == self._part_types[src_typ],
                    )

                    # Filter local particle handles to those of the destination type.
                    part_hndls = [
                        x for x in loc_obj.get_owned_part()[0]
                        if x.type == self._part_types[loc_typ]
                    ]
                    # Copy properties from source to local, element-wise.
                    for local, src in zip(part_hndls, part_slice.particles):
                        if prop_src == "dip" and prop_loc == "director":
                            # Normalize dipole to unit vector for director.
                            val = getattr(src, prop_src)
                            norm = np.linalg.norm(val,axis=1,keepdims=True)
                            val /= norm
                            setattr(local, prop_loc, val)
                        else:
                            setattr(local, prop_loc, getattr(src, prop_src))

    def rebind_sys(self, new_sys):
        ''' Rebind the simulation to a new espresso system handle. This must be called after loading a checkpoint, otherwise the gloabal scope and internal reference to espressomd System will not match
        :param new_sys: espressomd.System | Global scope system handle to bind to.
        :return: None
        '''

        logging.debug('identity of local system',id(self.sys))
        logging.debug('identity of loaded espresso system',id(new_sys))
        object.__setattr__(self, "sys", new_sys)
        logging.debug('identity of espresso system from rebind_sys',id(self.sys))
        logging.info('successfully rebound to new espresso handle after checkpoint load!')

    def api_agnostic_feature_check(self,feature_name):
        ret_val=None
        espresso_major_version=espressomd.version.major()
        try:
            if espresso_major_version==5:
                ret_val=espressomd.code_features.has_features(feature_name)
            elif espresso_major_version==4:
                ret_val=espressomd.has_features(feature_name)
            else:
                raise ValueError('This version of ESPResSo may not be supported!')
        except RuntimeError:
            logging.warning(f'feature check for {feature_name}, espresso version {espresso_major_version} failed with exception {sysos.exc_info()}')
            return False
        return ret_val

    def get_pos_ori_from_src(
    self,
    registered_objs,
    time_step: int = -1,
):

        """
        Load particle positions and orientations for a set of registered local objects
        from an external HDF5 source.

        This function opens the HDF5 file specified by ``self.src_path_h5``, selects the
        particle group matching the class of the first object in ``registered_objs``,
        verifies that all requested source type names (``self.pos_ori_src_type``) exist
        both in the local type map (``self.part_types``) and in the source data
        (by numeric ID), and then, for each local object, selects the subset of source
        particles connected to that object and filtered by the requested types.

        For each object, positions are read directly. Orientations are read from the
        ``director`` property when available; if it is absent, orientations are derived
        by normalizing the ``dip`` vectors. Zero-magnitude dip vectors are rejected.

        Parameters
        ----------
        registered_objs : iterable
            Local group instances (e.g., Filament objects) whose owned particles
            should be updated. All objects are assumed to belong to the same particle
            group (same class).
        time_step : int, optional
            Index of the source time step to read. Use ``-1`` to select the last
            available frame (default). The underlying selector must support negative
            indexing if ``-1`` is used.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Two lists, each with one entry per object in ``registered_objs``:
            ``(positions_per_obj, orientations_per_obj)``.

            - ``positions_per_obj[i]`` has shape ``(Ni, 3)`` with particle positions
            for the *i*-th object.
            - ``orientations_per_obj[i]`` has shape ``(Ni, 3)`` with unit orientation
            vectors (either the stored ``director`` or normalized ``dip``).

        Raises
        ------
        AssertionError
            If any requested type name in ``self.pos_ori_src_type`` is missing from
            ``self.part_types``, or if its corresponding numeric ID is not present in
            the source data for the selected group.
        ValueError
            If orientation must be inferred from ``dip`` and one or more dip vectors
            have zero (or nonpositive) magnitude.
        KeyError
            If HDF5/group lookups fail (e.g., missing datasets), depending on the
            behavior of the data selector.
        Exception
            Other exceptions may propagate from ``H5DataSelector`` or the predicate,
            and are logged via ``sysos.exc_info()`` in the orientation fallback path.

        Notes
        -----
        - Filtering is performed with a predicate equivalent to
        ``np.isin(subset.type, allowed_type_ids)`` where
        ``allowed_type_ids = [self.part_types[name] for name in self.pos_ori_src_type]``.
        - If ``director`` is not present, orientations are computed as
        ``dip / ||dip||`` with an explicit check against zero norms.
        - This method assumes ``self.src_params_set`` is ``True`` and that
        ``self.src_path_h5`` points to a readable HDF5 file.
        - Logging includes a summary of local and source type mappings and per-object
        load operations.

        See Also
        --------
        H5DataSelector.select_particles_by_object : Used to gather per-object subsets.
        """

        # Open the source HDF5 and select the data group matching the requested type.
        assert self.src_params_set==True, 'src_params_set must be set before calling this method'
        with  h5py.File(self.src_path_h5, "r") as src_file:
            src_data_grp = H5DataSelector(src_file, particle_group=registered_objs[0].__class__.__name__)

            # Discover the set of numeric type IDs present in the source for this group.
            all_src_types_numeric = np.unique(src_data_grp.type)
            requested_names = set(self.pos_ori_src_type)
            available_names = set(self._part_types.keys())

            # 1) every requested name must exist
            missing_names = requested_names - available_names
            assert not missing_names, (
                f"source type(s) {sorted(missing_names)} not found in "
                f"simulation part types {sorted(available_names)}"
            )
            # 2) the numeric ids for those names must exist in the source data
            requested_ids = {self._part_types[name] for name in requested_names}
            available_ids = set(all_src_types_numeric)

            missing_ids = requested_ids - available_ids
            assert not missing_ids, (
                "source data is missing type id(s): "
                f"{sorted(missing_ids)} "
                f"({[self._part_types.key_for(i) for i in sorted(missing_ids)]} by name) "
                f"not found in source data part types {sorted(available_ids)}"
            )
            logging.info(f"simulation contains types: {dict(self._part_types)}")

            positions_per_obj,ori_per_obj=[],[]
            for loc_obj in registered_objs:
                logging.info(
                    f"Loading data for {loc_obj.__class__.__name__}: {loc_obj.who_am_i} from SRC part type {self.pos_ori_src_type}."
                )
                # Select source particles at the requested time step that belong to this group instance (connectivity == loc_obj.who_am_i), with the correct pos_ori_src_type.
                allowed_types=[self._part_types[x] for x in self.pos_ori_src_type]
                part_slice = src_data_grp.timestep[time_step].select_particles_by_object(
                    object_name=loc_obj.__class__.__name__,
                    connectivity_value=loc_obj.who_am_i,
                    predicate=lambda subset: np.isin(subset.type, allowed_types),
                )
                positions_per_obj.append(part_slice.pos)    
                try:
                    ori_per_obj.append(part_slice.director)
                except KeyError:
                    exc_type, value, traceback = sysos.exc_info()
                    logging.debug("Failed with exception [%s,%s ,%s]" %
                        (exc_type, value, traceback))
                    val = part_slice.dip
                    norm = np.linalg.norm(val,axis=1,keepdims=True)
                    if np.any(norm==0.0):
                        raise ValueError(f"dip moment magnitude is 0 and cannot be used to infer particle orientation!")
                    val /= norm
                    ori_per_obj.append(val)
                    continue
        return positions_per_obj,ori_per_obj
import espressomd
import numpy as np
from collections import defaultdict
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.helper_functions import RoutineWithArgs, PartDictSafe, SinglePairDict, BondWrapper, get_neighbours, add_box_constraints_func, remove_box_constraints_func, check_free_cuboid, calculate_pair_distances, fcc_lattice, generate_random_unit_vectors, normalize_vectors, get_neighbours_ordered
import logging
import warnings
import os
import sys as sysos

class Elastomer(metaclass=Simulation_Object):
    '''
    Class that contains elastomer relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Elastomer. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    simulation_type=SinglePairDict('elastomer', 98)
    part_types = PartDictSafe({'real': 1, 'virt': 2, 'substrate': 98})
    config = ObjectConfigParams(
        box_E= None,
        box_E_shift= np.array((0.,0.,0.)),
        n_parts= None,
        bond_type= "HarmonicBond",
        bond_K_dist= "normal",
        bond_K_lims= (0.01,0.1),
        seed= int.from_bytes(os.urandom(2), sysos.byteorder)
        )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a elastomer object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        if config['box_E'] is None:
            config['box_E'] = self.sys.box_l
        config['box_E'] = np.asarray(config['box_E'])
        if config['n_parts'] is None:
            config['n_parts']= 0.3 * (np.prod(config['box_E']) /  4.18879)
            warnings.warn('monomer size assumed to be 1. and inferred number of particles from volume of Elastomer (to get 0.3 volume fraction)')
        self.params=config
        self.associated_objects=self.params['associated_objects']
        self.substrate=None
        self.build_function=RoutineWithArgs(
            func=self.build_Elastomer,
            num_monomers=self.params['n_parts']
            )
        self.who_am_i = Elastomer.numInstances
        Elastomer.numInstances += 1
        self.type_part_dict=PartDictSafe({key: [] for key in Elastomer.part_types.keys()})
        assert  Elastomer.numInstances <= 1, "It is only possible to have an elastomer at a time."

    def to_dict_of_god(self, father=True):
        raise NotImplementedError("Does not work for pickle'ing whole elastomer. May work for other purposes.")
        # Start with obj's own attributes (handle metaclass attributes carefully)
        if hasattr(self, 'get_params'):
            base_dict = self.get_params().copy()
            print("1", base_dict.keys())
        else:
            # fallback: serialize __dict__ if available
            base_dict = getattr(self, '__dict__', {}).copy()
            print("2", base_dict.keys())

        # Remove or transform attributes that cause problems (e.g., handles to espresso system)
        base_dict.pop('sys', None)
        base_dict.get('associated_objects', None)  # handle separately

        # Recursively serialize associated objects if present
        super_associated_objs = collect_objs_recursively(self)[1:]
        for obj in super_associated_objs:
            obj.to_dict_of_god(father=False)

        if father:
            for part_list in self.type_part_dict.values():
                for part in part_list:
                    part.to_dict_of_god()

    def set_object(self, pos, ori):
        '''
        Sets the particles in espresso according to self.build_funciton. Particles created here are treated according to their class set_object functions. Indices of added particles stored in self.realz_indices.append attribute.

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        pos=np.atleast_2d(pos)
        assert np.all((pos >= self.params['box_E_shift']) & (pos <= self.params['box_E_shift'] + self.params['box_E'])), "particle positions are outisde of elastomer space. Make sure to use elastomer.params['box_E'] and elastomer.params['box_E_shift'] as inputs to the Simulation.set_objects."
        assert check_free_cuboid(self.sys, self.params['box_E'], self.params['box_E_shift']), "Elastomer must be build on empty space. Adjust box_E and box_E_shift or remove non-elastomer particles to make space."
        assert len(
            pos) == self.params['n_parts'], 'there is a missmatch between the pos lenth and Elastomer n_parts'
        if self.associated_objects is None:
            dipm= 1.
            logic = (self.add_particle(type_name='real',pos=pp, dip=(dipm * oo), rotation=(True, True, True)) for pp, oo in zip(pos, ori))
        else:
            assert self.params['n_parts'] == len(
                self.associated_objects), " there doest seem to be enough particles stored!!! "
            if not all(hasattr(obj, 'set_object') and callable(getattr(obj, 'set_object')) for obj in self.associated_objects):
                raise TypeError("One or more objects do not implement a callable 'set_object'")
            logic = (obj_el.set_object(pos_el, ori_el)
                        for obj_el, pos_el, ori_el in zip(self.associated_objects, pos, ori))
        for part in logic:
            pass
        
        return self
    
    def set_steric(self, epsilon=1.):
        if self.associated_objects is None:
            self.sys.non_bonded_inter[self.part_types['real'], self.part_types['real']].wca.set_params(epsilon=epsilon, sigma=self.params['size'])
        else:
            for obj in self.associated_objects:
                obj.set_steric(epsilon=epsilon)
    
    def build_Elastomer(self, box_lengths, center=np.array([0,0,0]), sphere_radius=1., num_monomers=1, spacing=None, flag='rand'):
        box_lengths= np.asarray(box_lengths)
        scaling = 1.0
        
        # Adjust scaling until we have enough sphere centers
        while True:
            sphere_centers = fcc_lattice(radius=sphere_radius, volume_sides=self.params['box_E'], scaling_factor=scaling)
            volumes_to_fill=len(sphere_centers)
            logging.info('num_spheres_needed, num_spheres_got: %s', (num_monomers, volumes_to_fill))
            if  volumes_to_fill>= num_monomers:
                break
            scaling -= 0.1
        logging.info('scaling used: %s', scaling)

        # Center point distribution in box
        min_centers = np.min(sphere_centers, axis=0)
        max_centers = np.max(sphere_centers, axis=0)
        sphere_centers += self.params['box_E']/2 - (min_centers + max_centers)/2

        # Randomly shuffle the available centers and select the required number of centers
        take_index = np.arange(len(sphere_centers))
        if flag=='rand':
            np.random.shuffle(take_index)
        take_index = take_index[:num_monomers]
        sphere_centers=sphere_centers[take_index]

        orientations, points = build_function_generic(self, box_l=box_lengths, num_children=num_monomers, children_centers=sphere_centers, children_size=sphere_radius)

        return orientations, points
    
    def mix_elastomer_stuff(self, iter_multiplier=1, substrate=None, test=False):
        if isinstance(self, list):
            raise ValueError("Must be used on Elastomer object type")

        # add iniziatilation process, to get a nice random distribution before bonding
        if test:
            n_inter_0 = 10
            n_iter_1 = 0
        else:
            n_inter_0 = 100
            n_iter_1 = int(2000000 * iter_multiplier)

        old_time_step= float(self.sys.time_step)

        # Add temporary box particles
        types_M = tuple(typ for key, typ in self.part_types.items() if "real" in key)
        add_box_constraints_func(sides=['no-sides'], top=(self.params['box_E'][2] + self.params['box_E_shift'][2]), bottom=self.params['box_E_shift'][2], inter='wca', types_=types_M, sys=self.sys)

        # Make sure particles are not overlapping walls or substrate
        self.sys.integrator.set_steepest_descent(f_max=0, gamma=100, max_displacement=0.1)
        energy_non_bonded = self.sys.analysis.energy()['non_bonded']
        while energy_non_bonded > 0 and not test:
            self.sys.integrator.run(n_inter_0)
            energy_non_bonded = self.sys.analysis.energy()['non_bonded']
        self.sys.integrator.set_vv()

        # First relaxation (high T)
        self.sys.thermostat.set_langevin(kT=0.5, gamma=10, seed=self.params['seed'])
        self.sys.time_step = 0.0001
        self.sys.integrator.run(n_iter_1)

        # Remove temporary box particles
        remove_box_constraints_func(sys=self.sys)

        self.sys.thermostat.turn_off()
        self.sys.time_step = old_time_step
        
    
    def cure_elastomer(self, test=False):
        if isinstance(self, list):
            raise ValueError("Must be used on Elastomer object type")

        # add iniziatilation process, to get a nice random distribution before bonding
        if test:
            n_inter_0 = 10
        else:
            n_inter_0 = 100

        if self.substrate is not None:
            # Stuck the bottom layer particles to the z=R_M plane
            #  (restrict movement in z direction)
            assert ( isinstance(self.substrate, espressomd.constraints.ShapeBasedConstraint) and isinstance(self.substrate.shape, espressomd.shapes.Wall) ) \
             or ( isinstance(self.substrate, (espressomd.particle_data.ParticleList, espressomd.particle_data.ParticleSlice)) ) \
            , "substrate must be None, an espresso wall constraint, or a particle list. Use Elastomer.create_substrate to create valid substrate."         
            z_subs_tmp= self.params['box_E_shift'][2]
            if self.associated_objects is None:
                z_tmp = z_subs_tmp + 0.5
                for hndl in self.type_part_dict['real']:
                    if hndl.pos[2] < (z_tmp + 0.25): # chose at which heights to capture Ms
                        hndl.pos = [hndl.pos[0], hndl.pos[1], z_tmp]
                        hndl.fix = [False, False, True]                
            else:
                for obj in self.associated_objects:
                    r_M = obj.params['size'] / 2
                    z_tmp =  z_subs_tmp + r_M
                    for key, typ in obj.part_types.items():
                        if "real" in key:
                            hndls= obj.type_part_dict[typ]
                            for hndl in hndls:
                                if hndl.pos[2] < (z_tmp + r_M/4): # chose at which heights to capture Ms
                                    hndl.pos = [hndl.pos[0], hndl.pos[1], z_tmp]
                                    hndl.fix = [False, False, True]

            types_M = tuple(typ for key, typ in self.part_types.items() if "real" in key)
            add_box_constraints_func(sides=['no-sides'], top=(self.params['box_E'][2] + self.params['box_E_shift'][2]), bottom=self.params['box_E_shift'][2], inter='wca', types_=types_M, sys=self.sys)

            old_time_step= float(self.sys.time_step)

            # Relaxation with substrate (low T)
            self.sys.thermostat.set_langevin(kT=1E-3, gamma=100, seed=self.params['seed'])
            self.sys.time_step=0.001
            energy_non_bonded = self.sys.analysis.energy()['non_bonded']
            n_count=0
            while energy_non_bonded > 0 and n_count < 100 and not test:
                self.sys.integrator.run(n_inter_0)
                energy_non_bonded = self.sys.analysis.energy()['non_bonded']
                n_count+=1

            remove_box_constraints_func(sys=self.sys)
            self.sys.thermostat.turn_off()
            self.sys.time_step = old_time_step

        # Bond particles
        r_catch = 5.
        max_bonds = 6
        bond_k = self.params['bond_K_lims']
        dist = self.params['bond_K_dist']
        n_bonds_if_0 = 3
        r_catch_if_0 = 5

        if self.params['bond_type'] == "HarmonicBond":
            _, n_bonds_dict = self.random_harmonic_bonds(r_catch, bond_k, max_bonds, r_cut=-1, dist=dist, std_scaling=6)
            logging.info("Added most bonds")
            lonely_M = []
            for id, n_bonds in n_bonds_dict.items():
                if n_bonds == 0:
                    lonely_M.append(id)
            self.bond_to_neighbors(parts=self.sys.part.by_ids(lonely_M), n_nghb=n_bonds_if_0, bond_k=bond_k, r_cut=-1, r_catch=r_catch_if_0, dist=dist, std_scaling=6)

    def relax_langevin(self, iter_multiplier=1, kT=1E-3, gamma=10, time_step=None, test=False):
        if isinstance(self, list):
            raise ValueError("Must be used on Elastomer object type")

        # add iniziatilation process, to get a nice random distribution before bonding
        if test:
            n_iter_1 = 0
        else:
            n_iter_1 = int(2000000 * iter_multiplier)

        if time_step is not None:
            old_time_step= float(self.sys.time_step)
            self.sys.time_step = time_step

        self.sys.thermostat.set_langevin(kT=kT, gamma=gamma, seed=self.params['seed'])
        self.sys.integrator.run(n_iter_1)
        if time_step is not None:
            self.sys.time_step = old_time_step

    def save_gzip(self, file):
        line_save=""
        for key, value in self.params.items():
            if key not in ("associeated_objects", "espresso_instance"):
                line_save += f"{key} {value} "
        line_save = line_save[:-1] + "\n"
        file.write(line_save.encode())

        if self.associated_objects is not None:
            raise NotImplementedError("Will implement soon soon.")
            n_objects_lookup = len(self.associated_objects)
            line_save = f"{n_objects_lookup}\n"
            file.write(line_save.encode())
            for obj in self.associated_objects:
                config_copy = config.copy()
                type_saved = config_copy.pop('type')
                type_name_saved = config_copy.pop('type_name')
                line_save = f"{type_name_saved} {type_saved}"
                for key in sorted(config_copy.keys()):
                    if key in ('espresso_handle'):
                        continue
                    value = config_copy[key]
                    if isinstance(value, (list, tuple)):
                        str_value = ','.join(str(item) for item in value)
                        line_save += f" {key}:{str_value}"
                    else:
                        line_save += f" {key}:{value}"
                line_save += "\n"
                file.write(line_save.encode())
                parts_type_saved = self.sys.part.select(type=type_saved)
                n_parts_lookup = len(parts_type_saved)
                line_save = f"{n_parts_lookup}\n"
                file.write(line_save.encode())
                if type_name_saved == "SUBSTRATE":
                    for part in parts_type_saved:
                        line_save = f"{part.id} {part.pos[0]} {part.pos[1]} {part.pos[2]}"
                        line_save += f" {part.fix[0]} {part.fix[1]} {part.fix[2]}"
                        line_save += "\n"
                        file.write(line_save.encode())
                else:
                    for part in parts_type_saved:
                        line_save = f"{part.id} {part.pos[0]} {part.pos[1]} {part.pos[2]}"
                        line_save += f" {part.dip[0]} {part.dip[1]} {part.dip[2]}"
                        line_save += f" {part.fix[0]} {part.fix[1]} {part.fix[2]}"
                        for bond in part.bonds:
                            line_save += f" {bond[0].k},{bond[0].r_0},{bond[1]}"
                        line_save += "\n"
                        file.write(line_save.encode())
        else:
            for typ, hndls in self.type_part_dict.items():
                line_save = f"parts len({hndls}) {typ}\n"
                for part in hndls:
                    line_save += f"{part.id} {part.pos[0]} {part.pos[1]} {part.pos[2]}"
                    line_save += f" {part.dip[0]} {part.dip[1]} {part.dip[2]}"
                    line_save += f" {part.fix[0]} {part.fix[1]} {part.fix[2]}"
                    line_save += f" {part.virtual} {part.vs_relative[0]}"
                    for bond in part.bonds:
                        line_save += f" {bond[0].k},{bond[0].r_0},{bond[1]}"
                    line_save += "\n"
                file.write(line_save.encode())

    def add_anchors(self,type_keys='all'): #TO BE DONE FOR ROTATION CONSTRAINTS
        '''
        Adds virtual particles at top and bottom of a particle with size sigma, as given by their director (aligned with the dipole moments for magnetic particles).
        Logic firstly adds front anchors and then back anchors, so there is a consistent logic to track ids. Indices of particles added here are stored in self.fronts_indices/self.backs_indices attributes respectively.

        :None:

        '''
        if type_keys == 'all':
            type_keys = tuple(typ for key, typ in self.part_types.items() if "real" in key)
        if self.associated_objects is not None:
            check_type_keys_recursive(obj=self, type_keys=type_keys)
            raise NotImplementedError('add_anchors is still WIP for generic objects')
        else:
            self.fronts_indices=[]
            self.backs_indices=[]

            handles = self.type_part_dict['real']

            logic_front = ((self.add_particle(type_name='virt', pos=part.pos + 0.5 * self.params['size'] * part.director, rotation=(False, False, False)), part) for part in handles)
            logic_back  = ((self.add_particle(type_name='virt', pos=part.pos - 0.5 * self.params['size'] * part.director, rotation=(False, False, False)), part) for part in handles)

            for part_front, part in logic_front:
                part_front.vs_auto_relate_to(part)
                self.fronts_indices.append(part_front.id)

            for part_back, part in logic_back:
                part_back.vs_auto_relate_to(part)
                self.backs_indices.append(part_back.id)
            logging.info(f'anchors added for Elastomer particles: {self.who_am_i}')

    def random_harmonic_bonds(self, r_catch, bond_k=(0.001, 0.01), max_bonds=None, r_cut=-1, dist="normal", std_scaling=6):
        """
        Randomly generate harmonic bonds between particle pairs within elastomer

        This method creates harmonic (elastic) bonds between random pairs of particles that are within 
        a specified maximum bonding distance (`r_catch`). The spring constant `k` for each bond is sampled from a truncated normal distribution defined over the interval `bond_k`, centered at its midpoint, with standard deviation equal to `(bond_k[1] - bond_k[0]) / std_scaling`.

        Parameters:
            r_catch (float): Maximum distance allowed between two particles to form a bond.
            bond_k (float or tuple of float, optional): Elastic constant bounds for the bonds.
                If a single number is given, all bonds use that value. If a tuple is provided,
                values are sampled from a truncated normal distribution between the two values.
                Defaults to (0.001, 0.01).
            max_bonds (int, optional): Maximum number of bonds each particle can form. 
                Defaults to all of the allowed bonds.
            std_scaling (float, optional): Scaling factor used to control the spread (std deviation) 
                of the `k` distribution. Higher values yield a narrower distribution. Default is 6.

        Returns:
            tuple:
                - total_bonds (int): Total number of harmonic bonds created.
                - n_bonds_dict (dict): Dictionary mapping each particle type pair to the number of 
                bonds created for that pair.

        Notes:
            - Bonds will be stores in either of the two particles bonded. There is no way to predict this.
            - Assumes a cubic simulation box (all box dimensions must be equal).
            - Applies periodic boundary conditions (PBC) when computing distances.
            - Each bond is added symmetrically and uniquely per particle pair.
            - Requires `get_neighbours` to provide a mapping of nearby particle IDs.
        """
        if self.substrate is not None:
            old_periodicity = np.copy(self.sys.periodicity)
            self.sys.periodicity = [True, True, False]
        ids=[]
        if self.associated_objects != None:
            for obj in self.associated_objects:
                for key, typ in obj.part_types.items():
                    if "real" in key:
                        particles.extend([p.id for p in obj.type_part_dict[typ]])
        else:
            ids.extend([p.id for p in self.type_part_dict['real']])
        particles = self.sys.part.by_ids(ids)

        if max_bonds is None:
            max_bonds = len(particles)

        box_size= self.sys.box_l[0]
        assert all(ele == box_size for ele in self.sys.box_l[1:]), "method assumes cubic box for system PBC"

        if isinstance(bond_k, (float, int)):
            bond_k = [bond_k, bond_k]
        assert (len(bond_k)==2
                and bond_k[1]-bond_k[0]>=0
               ), "method assumes bond_k to be either a number, or an interval represented by a tuple of the form (min, max)"
        
        assert r_cut > r_catch or r_cut == -1, "r_cut must be larger than any bond lenght. (default -1)"
        
        if dist in ("normal", "norm", "gaussian", "gauss"):
            mean_tmp = ( bond_k[1] + bond_k[0] ) / 2
            std_tmp = ( bond_k[1] - bond_k[0] ) / std_scaling

            dist_func = np.random.normal
            dist_kwargs = {'loc': mean_tmp, 'scale': std_tmp}
        else:
            raise ValueError(f"Tried to use unsupported distribution for elastomer bond strenght: '{dist}'. Supported distributions: 'normal'.")

        n_bonds_dict= defaultdict(int)

        pair_dict = get_neighbours_ordered(particles.pos, box_size, r_catch, map_indices=particles.id)

        n_bonds= 0
        for particle in particles:
            id1 = particle.id

            bonds_per_HM = n_bonds_dict[id1]
            if bonds_per_HM >= max_bonds:
                continue
            
            # remove neighbors that already have max bonds
            pair_dict_safe = pair_dict[id1].copy()
            for id2 in pair_dict_safe:
                if n_bonds_dict[id2] >= max_bonds:
                    pair_dict[id1].remove(id2)

            ids2 = np.random.choice(pair_dict[id1], min(len(pair_dict[id1]), max_bonds-bonds_per_HM), replace=False)

            for id2 in ids2:

                assert len(self.sys.part.by_id(id2).bonds) < max_bonds, f"len{len(self.sys.part.by_id(id2).bonds)}, n_bond{n_bonds_dict[id2]}"

                particle_nghb = self.sys.part.by_id(id2)

                r_12 = self.sys.distance(p1=particle, p2=particle_nghb)

                k_12= bond_k[0] - 1
                while k_12<bond_k[0] or k_12>bond_k[1]:
                    k_12 = dist_func(**dist_kwargs)
                elastic_bond = espressomd.interactions.HarmonicBond(r_0=r_12, k=k_12, r_cut=r_cut)

                self.sys.bonded_inter.add(elastic_bond)
                self.sys.part.by_id(id1).add_bond((elastic_bond, id2))

                n_bonds_dict[id1] += 1; n_bonds_dict[id2] += 1 # keep count of n bonds
                pair_dict[id2].remove(id1) # remove already bonded pairs from pair_dict

                assert r_12<=r_catch

            n_bonds+= n_bonds_dict[id1]

        if self.substrate is not None:
            self.sys.periodicity = old_periodicity

        logging.info([(key, x) for key, x in n_bonds_dict.items() if x!=6])

        return sum(n_bonds_dict.values()), n_bonds_dict
    
    def bond_to_neighbors(self, parts, n_nghb=3, bond_k=(0.001,0.01), r_catch=None, r_cut=-1, dist="normal", std_scaling=6):
        """bond to n nearest neighbors, with elastic bond"""

        if self.substrate is not None:
            old_periodicity = np.copy(self.sys.periodicity)
            self.sys.periodicity = [True, True, False]

        box_size= self.sys.box_l[0]
        assert all(ele == box_size for ele in self.sys.box_l[1:]), "method assumes cubic box for system PBC"

        if isinstance(bond_k, (float, int)):
            bond_k = [bond_k, bond_k]
        assert (len(bond_k)==2
                and bond_k[1]-bond_k[0]>=0
               ), "method assumes bond_k to be either a number, or an interval represented by a tuple of the form (min, max)"
        
        if r_catch is None:
            r_catch = box_size/2
        assert r_cut > r_catch or r_cut == -1, "r_cut must be larger than any bond lenght. (default -1)"
        
        if dist in ("normal", "norm", "gaussian", "gauss"):
            mean_tmp = ( bond_k[1] + bond_k[0] ) / 2
            std_tmp = ( bond_k[1] - bond_k[0] ) / std_scaling

            dist_func = np.random.normal
            dist_kwargs = {'loc': mean_tmp, 'scale': std_tmp}
        else:
            raise ValueError(f"Tried to use unsupported distribution for elastomer bond strenght: '{dist}'. Supported distributions: 'normal'.")
        
        neighbours_dict = get_neighbours_ordered(parts.pos, box_size, r_catch, map_indices=parts.id)
        for id1 in parts.id:
            particle = self.sys.part.by_id(id1)
            n_count = 0
            for id2 in neighbours_dict[id1]:
                if n_count == n_nghb:
                    break

                particle_nghb = self.sys.part.by_id(id2)

                r_12 = self.sys.distance(p1=particle, p2=particle_nghb)

                mean_tmp = ( bond_k[1] + bond_k[0] ) / 2
                std_tmp = ( bond_k[1] - bond_k[0] ) / 6
                k_12= bond_k[0] - 1
                while k_12<bond_k[0] or k_12>bond_k[1]:
                    k_12 = dist_func(**dist_kwargs)
                elastic_bond = espressomd.interactions.HarmonicBond(r_0=r_12, k=k_12, r_cut=r_cut)

                self.sys.bonded_inter.add(elastic_bond)
                self.sys.part.by_id(id1).add_bond((elastic_bond, id2))

                assert r_12<=r_catch

                n_count+= 1

            assert n_count == n_nghb
        
        if self.substrate is not None:
            self.sys.periodicity = old_periodicity
    
    def add_volume_potencial(self): #TO BE DONE TO TRY TO GET INCOMPRESSIBILITY (looking into CGAL alpha shapes)
        flat_part_list=[]
        # if self.associated_objects is not None:
        #     for obj in self.associated_objects:
        #         flat_part_list.extend(obj.type_part_dict[type_key])
        # else:
        #     flat_part_list.extend(self.type_part_dict[type_key])

        # if bond_handle not in self.sys.bonded_inter:
        #     self.sys.bonded_inter.add(bond_handle)
        #     logging.info(f'bond handle added to system for Object {self.__class__,self.who_am_i}')

        # for iid in range(1,len(flat_part_list)-1):
        #     flat_part_list[iid].add_bond((bond_handle, flat_part_list[iid+1],  flat_part_list[iid-1]))

    def create_substrate(self, geometry: str = 'wall'):
        if geometry == 'wall':
            self.create_substrate_wall()
        else:
            self.create_substrate_part()

    def remove_substrate(self, geometry: str = 'wall'):
        if geometry == 'wall':
            self.remove_substrate_wall()
        else:
            self.remove_substrate_part()

    def create_substrate_part(self):
        n_substrate_x = int(np.ceil(self.params['box_E'][0]))
        n_substrate_y = int(np.ceil(self.params['box_E'][1]))
        n_substrate= n_substrate_x * n_substrate_y
        pos_x, pos_y = np.meshgrid( np.linspace(0.5, self.params['box_E'][0]-0.5, n_substrate_x),
                                    np.linspace(0.5, self.params['box_E'][1]-0.5, n_substrate_y) )
        pos = np.column_stack((pos_x.ravel(), pos_y.ravel(), ( np.zeros(n_substrate) - 0.5 + self.params['box_E_shift'][2] ) ))
        substrate = self.sys.part.add(type=[self.part_types['substrate']]*n_substrate, pos=pos, virtual=[True]*n_substrate, fix=[[True,True,True]]*n_substrate)
        for part in substrate:
            part.vs_relative = (part.id, 0.0, (1,0,0,0))
        self.substrate = substrate

        for key, typ in self.part_types.items():
            if "real" in key:
                sigma = (0.5 + self.sys.non_bonded_inter[typ,typ].wca.sigma/2) / 2**(1/6)
                if sigma < 0.001:
                    raise ValueError(f"Interaction of type {typ} with wall is 0, has these particles have no interaction defined. If you would like to have no interactions between particles, but only with wall, then hange this function or do it with normal espresso constraints.")
                self.sys.non_bonded_inter[self.part_types['substrate'], typ].wca.set_params(epsilon=1E6, sigma=sigma)
        
    def remove_substrate_part(self):
        self.substrate.remove()
        self.substrate = None

        for key, typ in self.part_types.items():
            if "real" in key:
                self.sys.non_bonded_inter[self.part_types['substrate'], typ].wca.deativate()
                assert not self.sys.non_bonded_inter[self.part_types['substrate'], typ].wca.is_active(), "error removing interactions, when removing elastomer substrate"


    def create_substrate_wall(self):
        types_M = tuple(typ for key, typ in self.part_types.items() if "real" in key)
        wall_constraints = add_box_constraints_func(bottom=self.params['box_E_shift'][2], wall_type=self.part_types['substrate'], inter='wca', types_=types_M, sys=self.sys)
        self.substrate = wall_constraints[0]
        
    def remove_substrate_wall(self):
        remove_box_constraints_func(wall_type=self.part_types['substrate'], sys=self.sys)
        self.substrate = None


# Make generic build funciton to recursively call objects build functions
def build_function_generic(parent_obj, box_l, num_children, children_centers, children_size, spacing=None):
        # copy from partition cuboid volume
        points_list = [None] * num_children
        orientations_list = [None] * num_children

        if parent_obj.associated_objects is None:
            children_orientations = random_like_nested_3d_vectors(children_centers)
            return children_orientations, children_centers
        else:
            children_handels = [obj for obj in parent_obj.associated_objects]

            assert np.all(box_l==box_l[0]), "this methods assumes cubic box for num_monomers > 1"
            box_length = box_l[0]
            grouped_positions = defaultdict(list)
            #grouped_volumes is a dictionary that contains all neighouring lattice sites sphere_diameter  
            grouped_volumes=get_neighbours(children_centers,volume_side=box_length,cuttoff=children_size)
            for i, (child_obj, center) in enumerate(zip(children_handels, children_centers)):
                valid_placement = False
                while not valid_placement:
                    orientation, points = child_obj.routine_per_volume(
                        center=center, num_monomers=child_obj.routine_per_volume.num_monomers, sphere_radius=(child_obj.params['size'] / 2), spacing=child_obj.routine_per_volume.spacing,
                        box_lengths=box_l)
                    should_proceed = True
                    
                    # Check for overlaps with points in neighboring spheres
                    for volume_id in grouped_volumes[i]:
                        if grouped_positions[volume_id]:
                            distances = calculate_pair_distances(points, grouped_positions[volume_id], box_lengths=box_l)
                            if np.any(distances <= child_obj.routine_per_volume.monomer_size):
                                should_proceed = False
                                break
                    
                    if should_proceed:
                        grouped_positions[i].extend(points)
                        points_list[i] = points
                        orientations_list[i] = orientation
                        valid_placement = True

            return orientations_list, points_list
        
def random_like_nested_3d_vectors(item, rng=None):
    """
    Recursively generate random 3D unit vectors,
    matching the shape of nested lists, tuples, or arrays.
    - Lists/tuples of len==3 with all numbers => treated as vector and replaced by a random unit vector.
    - NumPy arrays with last dimension == 3 => generate array of random unit vectors with same shape.
    - Otherwise recurse.
    Raises error if a scalar or other unsupported leaf is found.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if isinstance(item, (list, tuple)):
        # Check if it is a 3D vector (leaf)
        if len(item) == 3 and all(isinstance(x, (float, int)) for x in item):
            return generate_random_unit_vectors(1).flatten().tolist()
        else:
            return [random_like_nested_3d_vectors(sub, rng) for sub in item]
    
    elif isinstance(item, np.ndarray):
        if item.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3 for 3D vectors, got shape {item.shape}")
        
        n_vectors = np.prod(item.shape[:-1])
        vectors = generate_random_unit_vectors(n_vectors)
        vectors = vectors.reshape(item.shape)
        # Just to be sure normalize (your function is safe)
        return normalize_vectors(vectors, axis=-1)

    else:
        raise ValueError(f"Expected last dimension to be 3 for 3D vectors, got {item}")

def check_type_keys_recursive(obj, type_keys):
    if isinstance(type_keys, (tuple, list)):
        for typ in type_keys:
            assert typ in obj.part_types, f"Missing type_key '{typ}' in {obj}. Type key must exist in the part_types of all associated monomers!"
    else:
        assert type_keys in obj.part_types, f"Missing type_key '{type_keys}' in {obj}. Type key must exist in the part_types of all associated monomers!"

    if obj.associated_objects is not None:
        for child in obj.associated_objects:
            check_type_keys_recursive(child, type_keys)

def collect_objs_recursively(parents, keep_parents=True):
    """
    Traverse each obj in `parents` and return a flat preorder list
    of every object reachable via `.associated_objects`.
    Raises RuntimeError on any duplicate.
    """
    seen = set()
    result = []
    stack = list(parents)

    while stack:
        obj = stack.pop()
        if obj in seen:
            raise RuntimeError(f"Duplicate object detected: {obj!r}")
        seen.add(obj)
        result.append(obj)
        stack.extend(getattr(obj, "associated_objects", []) or [])

    return result
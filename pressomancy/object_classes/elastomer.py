import espressomd
import numpy as np
from collections import defaultdict
from itertools import combinations_with_replacement
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import RoutineWithArgs, partition_cuboid_volume, PartDictSafe, SinglePairDict, BondWrapper, get_neighbours
import logging
import warnings

class Elastomer(metaclass=Simulation_Object):
    '''
    Class that contains elastomer relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Elastomer. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=list()	
    numInstances = 0
    simulation_type=SinglePairDict('elastomer', 98)
    part_types = PartDictSafe({'real': 1, 'virt': 2})
    config = ObjectConfigParams(
        n_parts= None,
        bond_type= "HarmonicBond",
        bond_K_dist= "normal",
        bond_K_lims= (0.01,0.1)
        )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a elastomer object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        if config['n_parts'] is None:
            config['n_parts']= 0.3 * (self.sys.box_l**3 /  4.18879)
            warnings.warn('monomer size assumed to be 1. and inferred number of particles from volume of Elastomer (to get 0.3 volume fraction)')
        self.params=config
        self.associated_objects=self.params['associated_objects']
        self.build_function=RoutineWithArgs(
            func=partition_cuboid_volume,
            box_lenghts= self.sys.box_l,
            n_spheres= config['n_parts'],
            sphere_diameter= self.params['size'],
            flag= 'rand'
            )
        if self.associated_objects==None:
            self.build_function.sphere_diameter=1.
            warnings.warn('monomer size set to 1.')
        else:
            self.build_function.sphere_diameter=self.associated_objects[0].params['size']
        self.who_am_i = Elastomer.numInstances
        Elastomer.numInstances += 1
        self.type_part_dict=PartDictSafe({key: [] for key in Elastomer.part_types.keys()})

    def set_object(self,  pos, ori):
        '''
        Sets the particles in espresso according to self.build_funciton. Particles created here are treated according to their class set_object functions. Indices of added particles stored in self.realz_indices.append attribute.

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        pos=np.atleast_2d(pos)
        assert len(
            pos) == self.params['n_parts'], 'there is a missmatch between the pos lenth and Elastomer n_parts'
        if self.associated_objects is None:
            logic = (self.add_particle(type_name='real',pos=pp, rotation=(True, True, True), dip=oo) for pp, oo in (pos, ori))
        else:
            assert self.params['n_parts'] == len(
                self.associated_objects), " there doest seem to be enough partiles stored!!! "
            if not all(hasattr(obj, 'set_object') and callable(getattr(obj, 'set_object')) for obj in self.associated_objects):
                raise TypeError("One or more objects do not implement a callable 'set_object'")
            logic = (obj_el.set_object(pos_el, self.orientor)
                        for obj_el, pos_el in zip(self.associated_objects, pos))
        for part in logic:
            pass
        return self
    
    def cure_elastomer(self):
        # add functions to bond particles
        tmp=0

    def mix_elastomer_stuff(self):
        # add iniziatilation process, to get a nice random distribution before bonding
        tmp=0

    def add_anchors(self,type_key): #TO BE DONE FOR ROTATION CONSTRAINTS
        '''
        Adds virtual particles at top and bottom of a particle with size sigma, as determined by the orientation vector calculated by the get_orientation_vec(). Logic firstly adds front anchors and then back anchors, so there is a consistent logic to track ids. Indices of particles added here are stored in self.fronts_indices/self.backs_indices attributes respectively.

        :None:

        '''
        if self.associated_objects!=None:
            assert all([type_key in x.part_types.keys() for x in self.associated_objects]), 'type key must exist in the part_types of all associated monomers!'
            raise NotImplementedError('add_anchors is still WIP for generic objects')
        else:
            self.fronts_indices=[]
            self.backs_indices=[]

            handles = self.type_part_dict[type_key]
            director = self.orientor
            for pp in handles:
                pp.director = director
            logic_front = ((self.add_particle(type_name='virt', pos=pp.pos + 0.5 * self.params['sigma'] * director, rotation=(False, False, False)), pp) for pp in handles)
            logic_back = ((self.add_particle(type_name='virt', pos=pp.pos - 0.5 * self.params['sigma'] * director, rotation=(False, False, False)), pp) for pp in handles)

            for p_hndl_front, pp in logic_front:
                p_hndl_front.vs_auto_relate_to(pp)
                self.fronts_indices.append(p_hndl_front.id)

            for p_hndl_back, pp in logic_back:
                p_hndl_back.vs_auto_relate_to(pp)
                self.backs_indices.append(p_hndl_back.id)
            # logging.info(f'anchors added for Filament {self.who_am_i}')

    def random_harmonic_bonds(self, r_catch, bond_k=(0.001, 0.01), max_bonds=None, r_cut=-1, std_scaling=6):
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
        part_types = tuple(set(ele.part_types['real'] for ele in self.associatedd_objects))

        if max_bonds is None:
            max_bonds = len(self.sys.part.select(part_types))

        box_size= self.sys.box_l[0]
        assert all(ele == box_size for ele in self.sys.box_l[1:]), "method assumes cubic box for system PBC"

        if isinstance(bond_k, (float, int)):
            bond_k = [bond_k, bond_k]
        assert (len(bond_k)==2
                and bond_k[1]-bond_k[0]>=0
               ), "method assumes bond_k to be either a number, or an interval represented by a tuple of the form (min, max)"
        
        assert r_cut > r_catch or r_cut == -1, "r_cut must be larger than any bond lenght. (default -1)"

        n_bonds_dict= defaultdict(int)

        particles = self.sys.part.select(type=part_types)

        pair_dict = get_neighbours(particles.pos, box_size, r_catch, map_indices=particles.id)

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

                mean_tmp = ( bond_k[1] + bond_k[0] ) / 2
                std_tmp = ( bond_k[1] - bond_k[0] ) / std_scaling
                k_12= bond_k[0] - 1
                while k_12<bond_k[0] or k_12>bond_k[1]:
                    k_12 = np.random.normal(loc=mean_tmp, scale=std_tmp)
                elastic_bond = espressomd.interactions.HarmonicBond(r_0=r_12, k=k_12, r_cut=r_cut)

                self.sys.bonded_inter.add(elastic_bond)
                self.sys.part.by_id(id1).add_bond((elastic_bond, id2))

                n_bonds_dict[id1] += 1; n_bonds_dict[id2] += 1 # keep count of n bonds
                pair_dict[id2].remove(id1) # remove already bonded pairs from pair_dict

                assert r_12<=r_catch

            n_bonds+= n_bonds_dict[id1]

        return sum(n_bonds_dict.values()), n_bonds_dict
    
    def bond_to_neighbors(self, parts=None, n_nghb=3, bond_k=(0.001,0.01), r_catch=None, r_cut=-1, object_types=None, part_types=None):
        """bond to n nearest neighbors, with elastic bond"""
        part_types = tuple(set(ele.part_types['real'] for ele in self.associatedd_objects))

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
        
        if parts is None:
            parts = self.sys.part.select(part_types)
        
        neighbours_dict = get_neighbours(parts.pos, box_size, r_catch, map_indices=parts.id)
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
                    k_12 = np.random.normal(loc=mean_tmp, scale=std_tmp)
                elastic_bond = espressomd.interactions.HarmonicBond(r_0=r_12, k=k_12, r_cut=r_cut)
                assert k_12>=bond_k[0] and k_12<=bond_k[1]

                self.sys.bonded_inter.add(elastic_bond)
                self.sys.part.by_id(id1).add_bond((elastic_bond, id2))

                assert r_12<=r_catch

                n_count+= 1

            assert n_count == n_nghb
    
    def add_volume_potencial(self, type_key, bond_handle): #TO BE DONE TO TRY TO GET INCOMPRESSIBILITY
        flat_part_list=[]
        if self.associated_objects!=None:
            for obj in self.associated_objects:
                flat_part_list.extend(obj.type_part_dict[type_key])
        else:
            flat_part_list.extend(self.type_part_dict[type_key])

        if bond_handle not in self.sys.bonded_inter:
            self.sys.bonded_inter.add(bond_handle)
            logging.info(f'bond handle added to system for Object {self.__class__,self.who_am_i}')

        for iid in range(1,len(flat_part_list)-1):
            flat_part_list[iid].add_bond((bond_handle, flat_part_list[iid+1],  flat_part_list[iid-1]))

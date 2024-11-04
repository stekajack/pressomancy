import espressomd
import numpy as np
import random
from itertools import product
from pressomancy.object_classes.object_class import Simulation_Object 
from pressomancy.helper_functions import RoutineWithArgs, make_centered_rand_orient_point_array, PartDictSafe
from typing import Generic, TypeVar, Iterable

def rule_maker(choice_id, offset, n=3):
    top = [26, 45, 49, 30]
    bottom = [51, 70, 74, 55]
    length = 4
    trigger_warning = False

    try:
        i = bottom.index(choice_id-offset)
    except ValueError:
        i = top.index(choice_id-offset)
        trigger_warning = True

    results = []
    free_end = 0
    for _ in range(n):
        next_index = (i + 1) % length
        if trigger_warning:
            results.append((bottom[i]+offset, top[next_index]+offset))
            free_end = bottom[next_index]+offset
        else:
            results.append((top[i]+offset, bottom[next_index]+offset))
            free_end = top[next_index]+offset

        i = (i + 1) % length
    return results, free_end

class Filament(metaclass=Simulation_Object):
    '''
    Class that contains filament relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Filament. Therefore many relevant parameters are class specific, not instance specific.
    '''
    numInstances = 0
    sigma = 1
    part_types = PartDictSafe({'virt': 2, 'to_be_magnetized': 3})
    last_index_used = 0
    n_parts = 6
    bond_len = sigma*pow(2, 1/6.)+1-0.6
    fene_k = 40
    fene_r0 = 0
    fene_r_max = sigma*3.
    dip_magnitude = 0.
    size = 0.

    def __init__(self, sigma, n_parts, espresso_handle, monomers=None,size=None):
        '''
        Initialisation of a filament object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        assert isinstance(espresso_handle, espressomd.System)
        Filament.numInstances += 1
        Filament.sigma = sigma
        Filament.n_parts = n_parts
        Filament.sys = espresso_handle
        self.build_function=RoutineWithArgs(func=make_centered_rand_orient_point_array,num_monomers=self.n_parts)
        self.who_am_i = Filament.numInstances
        self.realz_indices = []
        self.fronts_indices = []
        self.backs_indices = []
        self.magnetizable_virts = []
        self.orientor = np.empty(shape=3, dtype=float)
        self.associated_monomers = monomers
        if size==None:
            Filament.size = Filament.sigma*Filament.n_parts
        else:
            Filament.size=size

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. When the StopIteration except is caught Filament.last_index_used += Filament.n_parts . Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        assert len(
            pos) == Filament.n_parts, 'there is a missmatch between the pos lenth and Filament n_parts'
        self.orientor = ori
        if self.associated_monomers is None:
            logic = ([Filament.sys.part.add(id=idpp+Filament.last_index_used, type=Filament.part_types['nonmagn'],
                                            pos=pp, rotation=(True, True, True)).id,] for idpp, pp in enumerate(pos, start=0))
        else:
            assert self.n_parts == len(
                self.associated_monomers), " there doest seem to be enough monomers stored!!! "
            if not all(hasattr(obj, 'set_object') and callable(getattr(obj, 'set_object')) for obj in self.associated_monomers):
                raise TypeError("One or more objects do not implement a callable 'set_object'")
            logic = (obj_el.set_object(pos_el, self.orientor)
                        for obj_el, pos_el in zip(self.associated_monomers, pos))

        while True:
            try:
                p_ids = next(logic)
                self.realz_indices.extend(p_ids)

            except StopIteration:
                Filament.last_index_used += Filament.n_parts
                break
        return self.realz_indices

    def add_anchors(self):
        '''
        Adds virtual particles at top and bottom of a particle with size sigma, as determined by the orientation vector calculated by the get_orientation_vec(). Logic firstly adds front anchors and then back anchors, so there is a consistent logic to track ids. Indices of particles added here are stored in self.fronts_indices/self.backs_indices attributes respectively.

        :None:

        '''
        handles = Filament.sys.part.by_ids(self.realz_indices)
        director = self.orientor
        for pp in handles:
            pp.director = director
        logic_front = ((Filament.sys.part.add(
            id=idpp + Filament.last_index_used, type=Filament.part_types['virt'], pos=pp.pos + 0.5 *
            Filament.sigma * director),
            pp) for idpp, pp in enumerate(handles, start=0))
        logic_back = ((Filament.sys.part.add(
            id=idpp + Filament.last_index_used, type=Filament.part_types['virt'], pos=pp.pos - 0.5 * Filament.sigma
            * director),
            pp) for idpp, pp in enumerate(handles, start=0))
        while True:
            try:
                p_hndl_front, pp = next(logic_front)
                p_hndl_front.vs_auto_relate_to(pp)
                self.fronts_indices.append(p_hndl_front.id)

            except StopIteration:
                Filament.last_index_used += Filament.n_parts
                print('last index of filament ' +
                      str((Filament.last_index_used, self.who_am_i)))
                break

        while True:
            try:
                p_hndl_back, pp = next(logic_back)
                p_hndl_back.vs_auto_relate_to(pp)
                self.backs_indices.append(p_hndl_back.id)
            except StopIteration:
                Filament.last_index_used += Filament.n_parts
                print('last index of filament ' +
                      str((Filament.last_index_used, self.who_am_i)))
                break

    def bond_overlapping_virtualz(self, bond_handle, crit=0.):
        '''
        Adds FENE bonds between virtuals that fulfill the crit distance criterion. In general, it is assumed that there are virtual anchors placed using the add_anchors() method, and that between two real parts one can always found a pair of either overlapping virts or at a distance corresponding to the FENE_r0 parameter (crit param can be arbitrary but should be realated to the aforementioned params). Relies on np.isclose().

        :return: None

        '''
        Filament.sys.bonded_inter.add(bond_handle)
        handles_font = list(Filament.sys.part.by_ids(self.fronts_indices))
        handles_back = list(Filament.sys.part.by_ids(self.backs_indices))

        logic = (pp_f.add_bond((bond_handle, pp_b)) for pp_f, pp_b in product(
            handles_font, handles_back) if np.isclose(np.linalg.norm(pp_f.pos-pp_b.pos), crit))
        while True:
            try:
                next(logic)
            except StopIteration:
                print('virts are bonded')
                break

    def add_dipolez_to_virts(self, dip_magnitude=1.):
        '''
        Adds virtual particles to the center of each particle whose index is stored in self.realz_indices. It is critical that said virtuals do not have a director and have disabled rotation!

        :param dip_magnitude: float | magnitude of the dipole moment to be asigned using the self.orientor unit vector. Default=1.
        :return: None

        '''
        Filament.dip_magnitude = dip_magnitude
        handles = Filament.sys.part.by_ids(self.realz_indices)
        logic = (
            (Filament.sys.part.add(
                id=idpp + Filament.last_index_used, type=Filament.part_types['to_be_magnetized'], pos=pp.pos,
                dip=Filament.dip_magnitude*self.orientor, rotation=(False, False, False)),
             pp) for idpp, pp in enumerate(handles, start=0))
        while True:
            try:
                p_hndl, pp = next(logic)
                p_hndl.vs_auto_relate_to(pp)
                self.magnetizable_virts.append(p_hndl.id)

            except StopIteration:
                Filament.last_index_used += Filament.n_parts
                print('last index of filament ' +
                      str((Filament.last_index_used, self.who_am_i)))
                break

    def add_dipolez_to_realz(self, dip_magnitude=1.):
        '''
        Adds dupoles tp real particels

        :param dip_magnitude: float | magnitude of the dipole moment to be asigned using the self.orientor unit vector. Default=1.
        :return: None

        '''
        Filament.dip_magnitude = dip_magnitude
        handles = Filament.sys.part.by_ids(self.realz_indices)
        for x in handles:
            x.dip = Filament.dip_magnitude*self.orientor

    def center_filament(self):
        list_parts = list(Filament.sys.part.by_ids(self.realz_indices))
        ref_index = int(self.n_parts*0.5)
        ref_pos = list_parts[ref_index].pos
        shift = ref_pos
        for elem in list_parts:
            elem.pos = elem.pos-shift
        Filament.sys.integrator.run(steps=0)
        print('center_filament() moved parts')

    def bond_center_to_center(self, bond_handle):
        Filament.sys.bonded_inter.add(bond_handle)
        for x in range(len(self.realz_indices)-1):
            self.sys.part.by_id(self.realz_indices[x]).add_bond(
                (bond_handle, self.realz_indices[x+1]))

    def bond_quadriplexes(self, bond_handle, mode='hinge'):
        '''
        associated_monomers contains monomer objects (assume quadriplex). We add cormer particles in each quadriplex pair to a pool of candidate corners: candidate1 and candidate2. Finaly checks which corner pairs have a distance Filament.sigma-2*fene_r0. Relies on np.isclose().
        :return: None

        '''
        Filament.sys.bonded_inter.add(bond_handle)
        monomer_pairs = zip(self.associated_monomers,
                            self.associated_monomers[1:])
        for pair in monomer_pairs:
            fene_r0 = pair[0].fene_r0
            candidate1, candidate2, pair_distances = [], [], []
            candidate1.extend(pair[0].associated_quartets[1].corner_particles)
            candidate1.extend(pair[0].associated_quartets[2].corner_particles)
            candidate2.extend(pair[-1].associated_quartets[1].corner_particles)
            candidate2.extend(pair[-1].associated_quartets[2].corner_particles)
            pos1 = np.array([x.pos for x in candidate1])
            pos2 = np.array([x.pos for x in candidate2])

            indices_int = np.arange(len(candidate1))
            index_combinations = np.array(
                list(product(indices_int, indices_int)))
            pair_distances = np.linalg.norm(
                pos1[index_combinations[:, 0]] - pos2[index_combinations[:, 1]], axis=-1)
            filtered_indices = index_combinations[np.isclose(
                pair_distances, Filament.sigma-2*fene_r0)]

            if mode == 'hinge':
                random_pair = random.choice(filtered_indices)
                self.sys.part.by_id(candidate1[random_pair[0]].id).add_bond(
                    (bond_handle, candidate2[random_pair[-1]].id))

            if mode == 'all':
                for pair in filtered_indices:
                    self.sys.part.by_id(candidate1[pair[0]].id).add_bond(
                        (bond_handle, candidate2[pair[-1]].id))

    def wrap_into_Tel(self, bond_handles):
        '''
        associated_monomers contains monomer objects (assume quadriplex). We add cormer particles in each quadriplex pair to a pool of candidate corners: candidate1 and candidate2. Finaly checks which corner pairs have a distance Filament.sigma-2*fene_r0. Relies on np.isclose().
        :return: None

        '''
        bond_handle, diag_bond = bond_handles
        Filament.sys.bonded_inter.add(bond_handle)
        Filament.sys.bonded_inter.add(diag_bond)
        for iid in range(len(self.associated_monomers)):
            monomer = self.associated_monomers[iid]
            fene_r0 = monomer.fene_r0
            candidates1 = []
            candidates1.extend(monomer.associated_quartets[1].corner_particles)
            candidates1.extend(monomer.associated_quartets[2].corner_particles)
            if monomer == self.associated_monomers[0]:
                start_part_id = random.choice(candidates1).id
            print('begin print', start_part_id)
            res, free_end = rule_maker(start_part_id, monomer.who_am_i*75)
            print(res, free_end)
            for id1, id2 in res:
                self.sys.part.by_id(id1).add_bond(
                    (diag_bond, id2))

            candidates2 = []

            try:
                monomer = self.associated_monomers[iid+1]
                candidates2.extend(
                    monomer.associated_quartets[1].corner_particles)
                candidates2.extend(
                    monomer.associated_quartets[2].corner_particles)
                candidate_pos = np.array([x.pos for x in candidates2])

                pair_distances = np.linalg.norm(
                    candidate_pos-self.sys.part.by_id(free_end).pos, axis=-1)
                filtered = np.isclose(
                    pair_distances, Filament.sigma-2*fene_r0)

                if filtered.any() == True:
                    index = np.argmax(filtered)
                    candidates2[index].add_bond(
                        (bond_handle, free_end))
                else:
                    print('alt filter')
                    pair_distances = np.linalg.norm(
                        candidate_pos-self.sys.part.by_id(start_part_id).pos, axis=-1)
                    filtered = np.isclose(
                        pair_distances, Filament.sigma-2*fene_r0)
                    index = np.argmax(filtered)
                    candidates2[index].add_bond(
                        (bond_handle, start_part_id))

                print(index, start_part_id)

                start_part_id = candidates2[index].id
                print('end print', start_part_id)
            except IndexError:
                print('end of chain rached')
                continue

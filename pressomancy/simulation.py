import espressomd
from espressomd.virtual_sites import VirtualSitesRelative
import espressomd.polymer
import sys as sysos
import numpy as np
import os
import gzip
import pickle
from itertools import product
from pressomancy.object_classes import *
from pressomancy.helper_functions import *

def partition_cubic_volume_oriented_rectangles(big_box_dim, num_spheres, small_box_dim, num_monomers):
    _, _, sphere_diameter = small_box_dim
    sphere_radius = sphere_diameter*0.5
    volumes_to_fill = 0

    x_partitions, y_partitions, z_partitions = (
        big_box_dim // small_box_dim).astype(int)

    x_len, y_len, z_len = small_box_dim
    x_coords = np.linspace(
        0.5 * x_len, big_box_dim[0] - 0.5 * x_len, x_partitions)
    y_coords = np.linspace(
        0.5 * y_len, big_box_dim[1] - 0.5 * y_len, y_partitions)
    z_coords = np.linspace(
        0.5 * z_len, big_box_dim[2] - 0.5 * z_len, z_partitions)

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    sphere_centers = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Adjust coordinates for partitions equal to 1
    if x_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 0] = big_box_dim[0] - 0.5 * x_len

    if y_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 1] = big_box_dim[1] - 0.5 * y_len

    if z_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 2] = big_box_dim[2] - 0.5 * z_len
    volumes_to_fill = len(sphere_centers)

    assert len(sphere_centers) >= num_spheres, \
        'must be enough possible volumes. intriduce a scaling factor'

    take_index = np.arange(len(sphere_centers))
    np.random.shuffle(take_index)
    take_index = take_index[:num_spheres]
    shift = sphere_radius / num_monomers
    alphas = np.linspace(-sphere_radius,
                         sphere_radius, num_monomers + 1)[:-1] + shift
    result = np.empty((num_spheres, num_monomers, 3))
    for i, iid in enumerate(take_index):
        center = sphere_centers[iid]
        theta = np.random.uniform(0, 2 * np.pi)
        phi = 0.
        x_points = center[0] + alphas * np.sin(phi) * np.cos(theta)
        y_points = center[1] + alphas * np.sin(phi) * np.sin(theta)
        z_points = center[2] + alphas * np.cos(phi)
        result[i] = np.column_stack((x_points, y_points, z_points))

    return sphere_centers[take_index], result

def singleton(aClass):
    def onCall(*args, **kwargs):
        if onCall.instance == None:
            onCall.instance = aClass(*args, **kwargs)
        return onCall.instance
    onCall.instance = None
    return onCall

@singleton
class Simulation():
    '''
    Singleton class that is intended to manage a suspension of objects stored in a class dict() attribute self.objects = {}. Initialisation of class wraps the espresso system handle and general suspension level attributes. Therefore espresso system needs to be instatiated beforehand. From there one can set the system, store objects and manage them either trough simulation level methods that in general loop trough stored objects and defers any real work to the object class method.

    '''
    volume_centers=[]
    volume_size=None
    part_positions=[]
    def __init__(self, n_part_tot, density, espresso_handle):
        assert isinstance(espresso_handle, espressomd.System)
        self.n_part_tot = n_part_tot
        self.density = density
        self.no_objects = 0
        self.objects = []
        self.n_parts_per_obj = 0
        self.part_types = {}
        self.seed = int.from_bytes(os.urandom(2), sysos.byteorder)
        self.sys = espresso_handle
        self.partitioned=None

        self.set_sys()

    def set_sys(self, timestep=0.01):
        '''
        Set espresso cellsystem params, and import virtual particle scheme. Run automatically on initialisation of the System class.
        '''
        np.random.seed(seed=self.seed)
        print('core.seed: ', self.seed)
        self.sys.periodicity = (True, True, True)
        self.sys.time_step = 0.005
        self.sys.cell_system.skin = 0.5
        # self.sys.min_global_cut = 3.
        self.sys.virtual_sites = VirtualSitesRelative(have_quaternion=False)
        assert type(self.sys.virtual_sites) is VirtualSitesRelative, 'VirtualSitesRelative must be set. If not, anything involving virtual particles will not work correctly, but it might be very hard to figure out why. I have wasted days debugging issues only to remember i commented out this line!!!'

    def store_objects(self, iterable_list):
        '''
        Method stores objects in the self.objects dict, if the object has a n_part and part_types attributes,
        and the list of objects passed to the method is commesurate with the system level attribute n_tot_parts.
        Populates the self.part_types attribute with types found in the objects that are stored.
        All objects that are stored should have the same types stored, but this is not checked explicitly
        '''
        assert all((hasattr(ob, 'n_parts') and hasattr(ob, 'part_types'))
                   for ob in iterable_list), "method requiers that stored objects have n_parts attribute"
        assert not set(iterable_list).intersection(self.objects), "Lists have common elements!"
        for element in iterable_list:
            self.objects.append(element)
            for key, val in zip(element.part_types.keys(), element.part_types.values()):
                self.part_types[key] = val
            self.no_objects += 1
        self.n_parts_per_obj = self.objects[0].n_parts
        print(np.shape(self.objects))
        pass

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
                objects), sphere_diameter=objects[0].size,routine_per_volume=objects[0].build_function)
                self.volume_centers.extend(centeres)
                self.part_positions.extend(positions)
                self.volume_size=objects[0].size
            else:
                centeres, positions, orientations = partition_cubic_volume(box_length=self.sys.box_l[0], num_spheres=len(
                objects), sphere_diameter=objects[0].size,routine_per_volume=objects[0].build_function)
                res=get_cross_lattice_noninterceting_volumes(centeres,objects[0].size,self.volume_centers,self.part_positions,self.volume_size,self.sys.box_l[0])
                mask=[key for key,val in res.items() if all(val)]
                positions=positions[mask]
                orientations=orientations[mask]

            logic = (object_el.set_object(pos_el, orient_el)
                    for object_el, pos_el, orient_el in zip(objects, positions, orientations))
            while True:
                try:
                    next(logic)
                except StopIteration:
                    print('Objects set!!!')
                    break

    def add_patches_triples(self):

        assert any(isinstance(ele, Quadriplex) for ele in self.objects), "method assumes simulation holds Quadriplex objects"
        # assert 'cation' not in self.part_types, f"Error: Key cation exists in the dictionary. add_patches_triples will not work correctly"

        self.part_types['patch'] = 3
        quadriplexes = [ele for ele in self.objects if isinstance(ele, Quadriplex)]
        for quad_el in quadriplexes:
            triples = quad_el.associated_quartets
            part_hndl_a = self.sys.part.add(type=3, pos=self.sys.part.by_id(
                triples[1].realz_indices[0]).pos, director=self.sys.part.by_id(triples[1].realz_indices[0]).director)
            part_hndl_a.vs_auto_relate_to(
                self.sys.part.by_id(triples[1].realz_indices[0]))

            part_hndl_b = self.sys.part.add(type=3, pos=self.sys.part.by_id(
                triples[2].realz_indices[0]).pos, director=self.sys.part.by_id(triples[2].realz_indices[0]).director)
            part_hndl_b.vs_auto_relate_to(
                self.sys.part.by_id(triples[2].realz_indices[0]))
            part_hndl_a.add_exclusion(part_hndl_b.id)

    def mark_for_collision_detection(self, object_type=Quadriplex, part_type=666):
        assert any(isinstance(ele, object_type) for ele in self.objects), "method assumes simulation holds correct type object"

        self.part_types['marked'] = 666
        objects_iter = [ele for ele in self.objects if isinstance(ele, object_type)]
        assert all((hasattr(ob, 'mark_covalent_bonds') and callable(getattr(ob, 'mark_covalent_bonds')))
                   for ob in objects_iter), "method requires that stored objects have mark_covalent_bonds() method"
        for obj_el in objects_iter:
            obj_el.mark_covalent_bonds(part_type=part_type)

    def init_magnetic_inter(self, actor_handle):
        print('direct summation magnetic interactions initiated')
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
        print('WCA interactions initiated')
        for key_el, key_el2 in product(key, key):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]
                                      ].wca.set_params(epsilon=wca_eps, sigma=sigma)

    def set_steric_custom(self, pairs=[(None, None),], wca_eps=[1.,], sigma=[1.,]):
        '''
        Custom WCA interactions setter that requoires each interaciton pair, epsilon and sigma to be specified .
        :param pairs: list of tuples of keys from self.part_types | Default only nonmagn WCA
        :param wca_epsilon: list of float | strength of the steric repulsion. 
        :param sigma: list of float | interaction range. 
        :return: None
        '''
        assert len(pairs) == len(wca_eps) and len(pairs) == len(
            sigma), 'epsilon and sigma must be specified explicitly for each type pair'
        print('WCA interactions initiated')
        for (key_el, key_el2), eps, sgm in zip(pairs, wca_eps, sigma):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]
                                      ].wca.set_params(epsilon=eps, sigma=sgm)

    def set_vdW(self, key=('nonmagn',), lj_eps=1., lj_size=1.):
        '''
        Set LJ central attraction between particles of types given in the key parameter.
        :param key: tuple of keys from self.part_types | Default only nonmagn LJ
        :param lj_epsilon: float | strength of the lj attraction. 

        :return: None

        Interaction length is allways determined from lj_size.
        '''
        lj_cut = 2.5*lj_size
        for key_el, key_el2 in product(key, key):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]].lennard_jones.set_params(
                epsilon=lj_eps, sigma=lj_size, cutoff=lj_cut, shift=0)
        print('vdW interactions initiated!')

    def set_vdW_custom(self, pairs=[(None, None),], lj_eps=[1.,], lj_size=[1.,]):

        assert len(pairs) == len(lj_eps) and len(pairs) == len(
            lj_size), 'epsilon and sigma must be specified explicitly for each type pair'
        for (key_el, key_el2), eps, sgm in zip(pairs, lj_eps, lj_size):
            lj_cut = 1.5*sgm
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]].lennard_jones.set_params(
                epsilon=eps, sigma=sgm, cutoff=lj_cut, shift=0)
        print('vdW interactions initiated!')

    def init_lb_GPU(self, kT, agrid, dens, visc, gamma, timestep=0.01):
        print('GPU LB method is beeing initiated')
        self.sys.thermostat.turn_off()
        self.sys.part[:].v = (0, 0, 0)
        lbf = espressomd.lb.LBFluidGPU(
            kT=1, seed=self.seed, agrid=agrid, dens=dens, visc=visc, tau=timestep)
        print(self.sys.actors.active_actors)
        if len(self.sys.actors.active_actors) == 2:
            self.sys.actors.remove(self.sys.actors.active_actors[-1])
        self.sys.actors.add(lbf)
        # gamma_MD = 1/(1/(6*np.pi*0.25*(args.eta))-1/((args.eta)*25))
        gamma_MD = gamma
        print('gamma_MD: '+str(gamma_MD))
        self.sys.thermostat.set_lb(
            LB_fluid=lbf, gamma=gamma_MD, seed=self.seed)
        print(lbf.get_params())
        print('GPU LB method is set with the params above.')
        return lbf

    def create_flow_channel(self, slip_vel=(0, 0, 0)):
        print("Setup LB boundaries.")
        top_wall = espressomd.shapes.Wall(normal=[1, 0, 0], dist=1) # type: ignore
        bottom_wall = espressomd.shapes.Wall( # type: ignore
            normal=[-1, 0, 0], dist=-(self.sys.box_l[0] - 1))

        top_boundary = espressomd.lbboundaries.LBBoundary( # type: ignore
            shape=top_wall, velocity=slip_vel)
        bottom_boundary = espressomd.lbboundaries.LBBoundary(shape=bottom_wall) # type: ignore

        self.sys.lbboundaries.add(top_boundary)
        self.sys.lbboundaries.add(bottom_boundary)

    def avoid_explosion(self, F_TOL, MAX_STEPS=10000, F_incr=100, I_incr=100):
        print('iterating with a force cap.')
        self.sys.integrator.run(0)
        while True:
            try:
                old_force = np.max(np.linalg.norm(
                    self.sys.part.all().f, axis=1))
                self.sys.force_cap = F_incr
                self.sys.integrator.run(I_incr)
                force = np.max(np.linalg.norm(self.sys.part.all().f, axis=1))
                rel_force = np.abs((force - old_force) / old_force)
                print(f'rel. force change: {rel_force:.2e}')
                if (rel_force < F_TOL) or (I_incr >= MAX_STEPS):
                    raise ValueError
                I_incr += I_incr
                F_incr += F_incr

            except ValueError:
                self.sys.force_cap = 0
                print('explosions avoided sucessfully!')
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

    def set_H_ext(self, H=(0, 0, 1.)):
        for x in self.sys.constraints:
            self.sys.constraints.remove(x)
            print('Removed old H')
        ExtH = espressomd.constraints.HomogeneousMagneticField(H=list(H))
        self.sys.constraints.add(ExtH)
        print('External field set: '+str(H))
        for x in self.sys.constraints:
            print(x)

    def get_H_ext(self):
        for x in self.sys.constraints:
            print(str(x.H))
            return x.H

    def init_pickle_dump(self, path_to_dump):
        dict_of_god = {}
        f = gzip.open(path_to_dump, 'wb')
        pickle.dump(dict_of_god, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return path_to_dump, 0

    def load_pickle_dump(self, path_to_dump):
        f = gzip.open(path_to_dump, 'rb')
        dict_of_god = pickle.load(f)
        f.close()
        return path_to_dump, int(list(dict_of_god.keys())[-1].split('_')[-1])+1

    def dump_to_init(self, path_to_dump, dungeon_witch_list, cnt):
        f = gzip.open(path_to_dump, 'rb')
        dict_of_god = pickle.load(f)
        f.close()
        dict_of_god['timestep_%s' % cnt] = [x.to_dict_of_god()
                                            for x in dungeon_witch_list]
        f = gzip.open(path_to_dump, 'wb')
        pickle.dump(dict_of_god, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def generate_positions(self, min_distance):
        object_positions = []
        while len(object_positions) < self.no_objects:
            new_position = np.random.random(3) * self.sys.box_l
            if all(np.linalg.norm(new_position - pos) >= min_distance for pos in self.sys.part.all().pos):
                if all(np.linalg.norm(new_position - existing_position) >= min_distance for existing_position in object_positions):
                    object_positions.append(new_position)
            print('position casing progress: ', len(
                object_positions)/self.no_objects)

        return np.array(object_positions)

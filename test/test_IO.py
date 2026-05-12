import numpy as np
import espressomd
from create_system import sim_inst , BaseTestCase
from pressomancy.simulation import Filament, Quartet, Quadriplex, Crowder, Elastomer, PointDipolePermanent
from pressomancy.helper_functions import BondWrapper
from pressomancy.analysis import H5DataSelector, H5ObservableSelector
import h5py
import tempfile
import os
from pressomancy.helper_functions import MissingFeature
import logging
import shutil
from unittest.mock import patch
import pressomancy.simulation as simulation_module
import warnings

class cestica():
    pass

def capture_particle_snapshot(parts, custom_prop=None):
    snap=[]
    for part in parts:
        new=cestica()
        for prop, _ in sim_inst.io_dict["properties"]:
            setattr(new, prop, getattr(part, prop))
        if custom_prop is not None:
            setattr(new, custom_prop, getattr(part, custom_prop))
        snap.append(new)
    return snap

def check_prop_dim(dataview, ref_parts, prop_shape, time_slice=-1, expected_types=None):
    prop, shape = prop_shape
    property_data_h5df=getattr(dataview,prop)
    property_data=[]
    time_part_slice=np.atleast_2d(ref_parts if time_slice is None else ref_parts[time_slice])
    for snap in time_part_slice:
        if expected_types is not None:
            property_data.append([getattr(part,prop) for part in snap if part.type in expected_types])
        else:
            property_data.append([getattr(part,prop) for part in snap])
    if shape == 1:
        property_data_h5df=np.squeeze(property_data_h5df, axis=-1)
    assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'

def get_and_check_complete_object(dataview, object_grp_name, identity, ref_parts, expected_types, time_slice):

    selection_source = dataview if time_slice is None else dataview.timestep[time_slice]
    selection=selection_source.select_particles_by_object(object_name=object_grp_name, connectivity_value=identity)
    for prop,shape in sim_inst.io_dict['properties']:
        check_prop_dim(selection, ref_parts, (prop, shape), time_slice=time_slice, expected_types=expected_types)
    for predicate_type in expected_types:
        selection=selection_source.select_particles_by_object(object_name=object_grp_name, connectivity_value=identity,predicate=lambda p:p.type==predicate_type)
        for prop,shape in sim_inst.io_dict['properties']:
            check_prop_dim(selection, ref_parts, (prop, shape), time_slice=time_slice, expected_types=[predicate_type])

class CommonH5DataSelectorTests:

    runner_script="repo/project/script.py"
    runner_script_repo ="main@abc1234-dirty"
    library_vers= "main@def5678"
    lib_path="/some/path/"
    author="dungeonwitch"
    email='dungeonwitch@dungeon.com'

    @classmethod
    def setUpClass(cls):
        sim_inst.set_author(cls.author, cls.email)
        super().setUpClass()
        sim_inst.sys.box_l = cls.box_dim
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.h5_filename = os.path.join(cls.tmpdir.name, "testfile.h5")
        cls.written_steps = [10, 20, 30, 40]
        cls.written_times = []
        cls.build_fixture()
        cls.part_snapshots = {group_type.__name__: [] for group_type in cls.group_types}
        if hasattr(cls, "observable_name"):
            cls.observable_value = np.zeros(3, dtype=np.float64)
            cls.observable_values = []
        with patch.object(simulation_module, "get_submission_creator_info", return_value=(cls.runner_script, cls.runner_script_repo)), patch.object(simulation_module, "get_repo_context", return_value=(cls.lib_path, cls.library_vers)):
            sim_inst.inscribe_part_group_to_h5(group_type=cls.group_types, h5_data_path=cls.h5_filename)
            if hasattr(cls, "observable_name"):
                sim_inst.inscribe_observable_group_to_h5(
                    observable_defs=[(cls.observable_name, 3, np.float64, cls.observable_value)],
                    h5_data_path=cls.h5_filename,
                    mode='NEW',
                )
        for frame_index, GLOBAL_COUNTER in enumerate(cls.written_steps):
            sim_inst.sys.integrator.run(1)
            if hasattr(cls, "observable_name"):
                cls.observable_value[:] = np.array([GLOBAL_COUNTER, frame_index + 1, -GLOBAL_COUNTER], dtype=np.float64)
                cls.observable_values.append(cls.observable_value.copy())
            sim_inst.write_registered_to_h5(time_step=GLOBAL_COUNTER)
            cls.written_times.append(sim_inst.sys.time)
            for group_type in cls.group_types:
                parts = []
                for obj in sim_inst.objects:
                    if isinstance(obj, group_type):
                        owned_parts, _ = obj.get_owned_part()
                        parts.extend(owned_parts)
                cls.part_snapshots[group_type.__name__].append(capture_particle_snapshot(parts))
        cls.reset_io_state()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir"):
            cls.tmpdir.cleanup()
            cls.tmpdir = None
        cls.reset_io_state()
        BaseTestCase.cleanup()
        assert len(sim_inst.sys.part) == 0
        super().tearDownClass()

    def tearDown(self):
        self.reset_io_state()
        super().tearDown()

    @staticmethod
    def reset_io_state():
        h5_file = sim_inst.io_dict.get("h5_file")
        if h5_file is not None:
            h5_file.flush()
            h5_file.close()
        sim_inst.io_dict["h5_file"] = None
        sim_inst.io_dict["flat_part_view"].clear()
        sim_inst.io_dict["registered_observables"] = {}
        sim_inst.io_dict["registered_group_type"] = None

    @staticmethod
    def check_box_data(dataview, expected_edges, expected_boundary=("periodic", "periodic", "periodic")):
        box = dataview.get_box()
        expected_edges = np.array(expected_edges, dtype=float, copy=True)
        np.testing.assert_equal(box["dimension"], len(expected_edges), err_msg="Box dimension from selector does not match!")
        np.testing.assert_equal(box["boundary"], expected_boundary, err_msg="Box boundary from selector does not match!")
        np.testing.assert_allclose(box["edges"], expected_edges, err_msg="Box edges from selector do not match!")

    def check_version_signing(self, dataview):
        np.testing.assert_array_equal(dataview.metadata["h5md"]["_meta"]["attributes"]["version"], np.array([1, 0], dtype=np.int32))
        self.assertEqual(dataview.metadata["h5md"]["creator"]["_meta"]["attributes"]["name"], self.runner_script)
        self.assertEqual(dataview.metadata["h5md"]["creator"]["_meta"]["attributes"]["version"], self.runner_script_repo)
        self.assertEqual(dataview.metadata["parameters"]["pressomancy"]["_meta"]["attributes"]["version"], self.library_vers)
        expected_part_types = {
            key: int(value)
            for key, value in sim_inst.part_types.items()
            if isinstance(value, (int, np.integer))
        }
        observed_part_types = {
            key: int(value)
            for key, value in dataview.metadata["parameters"]["pressomancy"]["part_types"]["_meta"]["attributes"].items()
        }
        self.assertEqual(observed_part_types, expected_part_types)
        self.assertEqual(dataview.metadata["h5md"]["author"]["_meta"]["attributes"]["name"], self.author)
        self.assertEqual(dataview.metadata["h5md"]["author"]["_meta"]["attributes"]["email"], self.email)

    @staticmethod
    def get_and_check_connectivity_predicate(dataview, object_grp_name, particle_type, control_ids):
        selected_ids = np.array(
            dataview.get_connectivity_values(
                object_grp_name,
                predicate=lambda subset: np.all(subset.timestep[-1].type == particle_type),
            ),
            dtype=int,
        )
        np.testing.assert_array_equal(control_ids, selected_ids, err_msg=f"{object_grp_name} predicate-filtered IDs do not match!")

    def check_expected_metadata(self, dataview, h5_file, particle_group=None):

        def metadata_node(metadata, path):
            node = metadata
            for key in path.split("/"):
                node = node[key]
            return node

        metadata = dataview.metadata
        particle_group = next(iter(h5_file["particles"])) if particle_group is None else particle_group
        self.assertEqual(metadata["_meta"]["type"], "Group")
        self.assertEqual(set(metadata["_meta"]["members"]), set(h5_file.keys()))
        for group_name in ("h5md", "parameters", "particles", "connectivity"):
            self.assertIn(group_name, h5_file)
            self.assertIn(group_name, metadata)
        if hasattr(self, "observable_name"):
            self.assertIn("observables", h5_file)
            self.assertIn("observables", metadata)

        group_paths = [
            "h5md",
            "parameters/pressomancy",
            f"particles/{particle_group}",
            f"connectivity/{particle_group}",
        ]
        if hasattr(self, "observable_name"):
            group_paths.append(f"observables/{self.observable_name}")
        dataset_paths = [f"particles/{particle_group}/id/value",
                         f"particles/{particle_group}/pos/value",
                         f"particles/{particle_group}/box/edges",
                         *(f"connectivity/{particle_group}/{dataset_name}" for dataset_name in h5_file[f"connectivity/{particle_group}"]),]
        if hasattr(self, "observable_name"):
                dataset_paths.extend([f"observables/{self.observable_name}/step", f"observables/{self.observable_name}/time", f"observables/{self.observable_name}/value"])

        for group_path in group_paths:
            h5_group = h5_file[group_path]
            meta_node = metadata_node(metadata, group_path)
            self.assertEqual(meta_node["_meta"]["type"], "Group")
            self.assertEqual(set(meta_node["_meta"]["members"]), set(h5_group.keys()))
            actual_attrs = {key: h5_group.attrs[key] for key in h5_group.attrs}
            observed_attrs = meta_node["_meta"]["attributes"]
            self.assertEqual(set(observed_attrs), set(actual_attrs))
            for key, value in actual_attrs.items():
                np.testing.assert_equal(observed_attrs[key], value)

        for dataset_path in dataset_paths:
            h5_dataset = h5_file[dataset_path]
            meta_node = metadata_node(metadata, dataset_path)
            self.assertEqual(meta_node["type"], "Dataset")
            self.assertEqual(meta_node["shape"], h5_dataset.shape)
            self.assertEqual(meta_node["dtype"], str(h5_dataset.dtype))
            actual_attrs = {key: h5_dataset.attrs[key] for key in h5_dataset.attrs}
            observed_attrs = meta_node["attributes"]
            self.assertEqual(set(observed_attrs), set(actual_attrs))
            for key, value in actual_attrs.items():
                np.testing.assert_equal(observed_attrs[key], value)

    def test_observables(self):
        if not hasattr(self, "observable_name"):
            return
        with h5py.File(self.h5_filename, "r") as h5_file:
            selector = H5ObservableSelector(h5_file, observable_name=self.observable_name)
            np.testing.assert_equal(len(selector.timestep), len(self.written_steps), err_msg="Observable selector length is incorrect!")
            np.testing.assert_array_equal(selector.step, self.written_steps, err_msg="Observable frame counters are incorrect!")
            np.testing.assert_allclose(selector.time, self.written_times, err_msg="Observable times do not match fixture write times.")
            np.testing.assert_allclose(selector.value, np.array(self.observable_values), err_msg="Observable values do not match fixture payloads.")
            sliced = selector.timestep[0:2]
            np.testing.assert_equal(len(sliced.timestep), 2, err_msg="Observable timestep slicing did not preserve frame count.")
            np.testing.assert_array_equal(sliced.step, self.written_steps[0:2], err_msg="Observable timestep slicing did not preserve steps.")
            np.testing.assert_allclose(sliced.time, self.written_times[0:2], err_msg="Observable timestep slicing did not preserve times.")
            np.testing.assert_allclose(sliced.value, np.array(self.observable_values[0:2]), err_msg="Observable timestep slicing did not preserve values.")
            frames = [frame for frame in selector.timestep]
            np.testing.assert_equal(len(frames), len(self.written_steps), err_msg="Observable timestep iteration did not yield every frame.")
            np.testing.assert_array_equal([frame.step for frame in frames], self.written_steps, err_msg="Observable timestep iteration did not preserve steps.")

    def test_metadata(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                particle_group = group_type.__name__
                expected_particles = 0
                for obj in sim_inst.objects:
                    if isinstance(obj, group_type):
                        parts, _ = obj.get_owned_part()
                        expected_particles += len(parts)
                dataview = H5DataSelector(h5_file, particle_group=particle_group)
                self.check_box_data(dataview, self.box_dim)
                self.check_version_signing(dataview)
                self.check_expected_metadata(dataview, h5_file, particle_group=particle_group)
                expected_timesteps = len(self.written_steps)
                np.testing.assert_equal(
                    dataview.common_dims,
                    (expected_timesteps, expected_particles),
                    err_msg=f"{particle_group} selector common dimensions do not match!",
                )
                np.testing.assert_equal(len(dataview.timestep), expected_timesteps, err_msg=f"{particle_group} timestep length does not match!")
                np.testing.assert_equal(len(dataview.particles), expected_particles, err_msg=f"{particle_group} particle count does not match!")

    def test_trigger_exceptions_smoke(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            dataview = H5DataSelector(
                h5_file, particle_group=self.group_types[0].__name__
                )
            tests = [
                (lambda: H5DataSelector(h5_file, particle_group="DangerNoodle"), ValueError),
                (lambda: dataview[-1], TypeError),
                (lambda: iter(dataview), TypeError),
                (lambda: len(dataview), TypeError),
            ]
            if hasattr(self, "observable_name"):
                observable_selector = H5ObservableSelector(h5_file, observable_name=self.observable_name)
                tests.extend([
                    (lambda: H5ObservableSelector(h5_file, observable_name="missing_observable"), ValueError),
                    (lambda: observable_selector[0], TypeError),
                    (lambda: iter(observable_selector), TypeError),
                    (lambda: len(observable_selector), TypeError),
                ])
            for fn, exc in tests:
                with self.assertRaises(exc):
                    fn()

    def test_mk_src_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            dst_filename = os.path.join(tmpdirname, "dst.h5")
            sim_inst.mk_src_file(self.h5_filename, dst_filename)

            with h5py.File(dst_filename, "r") as h5_file:
                for group_type in self.group_types:
                    particle_group = group_type.__name__
                    dataview = H5DataSelector(h5_file, particle_group=particle_group)
                    self.check_box_data(dataview, self.box_dim)
                    self.check_version_signing(dataview)
                    self.check_expected_metadata(dataview, h5_file, particle_group=particle_group)
                    np.testing.assert_array_equal(dataview.step, np.array([self.written_steps[-1]], dtype=np.int32))
                    check_prop_dim(dataview, [self.part_snapshots[particle_group][-1]], ("pos", 3), time_slice=0)

    def test_load_modes(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for mode in ("LOAD_NEW", "LOAD"):
                h5_filename = os.path.join(tmpdirname, f"{mode}.h5")
                shutil.copy2(self.h5_filename, h5_filename)
                saved_part_types = dict(sim_inst.part_types)
                try:
                    if mode == "LOAD_NEW":
                        sim_inst.part_types.clear()
                    GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
                        group_type=self.group_types,
                        h5_data_path=h5_filename,
                        mode=mode,
                    )
                    self.assertEqual(GLOBAL_COUNTER, len(self.written_steps))
                    if mode == "LOAD_NEW":
                        self.assertEqual(dict(sim_inst.part_types), saved_part_types)
                    for group_type in self.group_types:
                        group_name = group_type.__name__
                        expected_ids = []
                        for obj in sim_inst.objects:
                            if isinstance(obj, group_type):
                                parts, _ = obj.get_owned_part()
                                expected_ids.extend(part.id for part in parts)
                        reconstructed_ids = [part.id for part in sim_inst.io_dict['flat_part_view'][group_name]]
                        np.testing.assert_array_equal(
                            reconstructed_ids,
                            expected_ids,
                            err_msg=f"{mode} flat_part_view for {group_name} does not match live object order.",
                        )
                    if hasattr(self, "observable_name"):
                        observable_counter = sim_inst.inscribe_observable_group_to_h5(
                            observable_defs=[(self.observable_name, self.observable_value.shape, self.observable_value.dtype, self.observable_value)],
                            h5_data_path=h5_filename,
                            mode=mode,
                        )
                        self.assertEqual(observable_counter, len(self.written_steps))
                        registered = sim_inst.io_dict['registered_observables']
                        self.assertIn(self.observable_name, registered)
                        self.assertEqual(registered[self.observable_name]['shape'], self.observable_value.shape)
                        self.assertEqual(registered[self.observable_name]['dtype'], self.observable_value.dtype)
                        self.assertIs(registered[self.observable_name]['value'], self.observable_value)
                        selector = H5ObservableSelector(sim_inst.io_dict['h5_file'], observable_name=self.observable_name)
                        np.testing.assert_array_equal(selector.step, self.written_steps)
                        np.testing.assert_allclose(selector.time, self.written_times)
                        np.testing.assert_allclose(selector.value, np.array(self.observable_values))
                finally:
                    sim_inst.part_types.clear()
                    sim_inst.part_types.update(saved_part_types)
                    self.reset_io_state()

    def test_load_modes_force_resize(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for mode in ("LOAD_NEW", "LOAD"):
                h5_filename = os.path.join(tmpdirname, f"{mode}_resized.h5")
                shutil.copy2(self.h5_filename, h5_filename)
                GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
                    group_type=self.group_types,
                    h5_data_path=h5_filename,
                    mode=mode,
                    force_resize_to_size=2,
                )
                self.assertEqual(GLOBAL_COUNTER, 2)
                for group_type in self.group_types:
                    particle_group = group_type.__name__
                    dataview = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group=particle_group)
                    np.testing.assert_array_equal(dataview.step,
                                                  self.written_steps[:2])
                    np.testing.assert_allclose(dataview.time,
                                               self.written_times[:2])
                    np.testing.assert_equal(dataview.common_dims, (2, len(self.part_snapshots[particle_group][0])))

                if hasattr(self, "observable_name"):
                    observable_counter = sim_inst.inscribe_observable_group_to_h5(
                        observable_defs=[(self.observable_name, self.observable_value.shape, self.observable_value.dtype, self.observable_value)],
                        h5_data_path=h5_filename,
                        mode=mode,
                        force_resize_to_size=2,
                    )
                    selector = H5ObservableSelector(sim_inst.io_dict['h5_file'], observable_name=self.observable_name)
                    self.assertEqual(observable_counter, 2)
                    np.testing.assert_array_equal(selector.step, self.written_steps[:2])
                    np.testing.assert_allclose(selector.time, self.written_times[:2])
                    np.testing.assert_allclose(selector.value,
                    np.array(self.observable_values[:2]))
                self.reset_io_state()

    def test_select_particles_by_object(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                object_name = group_type.__name__
                dataview = H5DataSelector(h5_file, particle_group=object_name)
                np.testing.assert_equal(len(dataview.timestep), len(self.written_steps), err_msg=f"{object_name} stored frame count does not match fixture writes.")
                self.assertEqual(dataview.timestep[-1].step, self.written_steps[-1])
                self.assertEqual(dataview.timestep[-1].time, self.written_times[-1])
                objects = [obj for obj in sim_inst.objects if isinstance(obj, group_type)]
                connectivity_value = np.array([obj.who_am_i for obj in objects], dtype=int)
                expected_types = sorted({
                    int(part.type)
                    for obj in objects
                    for part in obj.get_owned_part()[0]
                })
                ids = dataview.get_connectivity_values(object_name)
                np.testing.assert_array_equal(ids, connectivity_value, err_msg=f"{object_name} connectivity IDs do not match fixture object ids!")
                connected_objects = sim_inst._collect_instances_recursively(objects)
                connected_object_names = sorted({obj.__class__.__name__ for obj in connected_objects})
                for connected_object_name in connected_object_names:
                    connected_class_objects = [obj for obj in connected_objects if obj.__class__.__name__ == connected_object_name]
                    for predicate_type in expected_types:
                        control_ids = [obj.who_am_i for obj in connected_class_objects if all(part.type == predicate_type for part in obj.get_owned_part()[0])]
                        self.get_and_check_connectivity_predicate(dataview, connected_object_name, predicate_type, control_ids)
                for time_slice in [None, -1, 0, slice(0, 2, 1)]:
                    get_and_check_complete_object(dataview, object_name, connectivity_value, self.part_snapshots[object_name], expected_types, time_slice=time_slice)
                for predicate_type in expected_types:
                    selection = dataview.select_particles_by_object(
                        object_name=object_name,
                        connectivity_value=connectivity_value,
                        predicate=lambda subset, predicate_type=predicate_type: subset.timestep[-1].type == predicate_type,
                    )
                    np.testing.assert_equal(len(selection.timestep), len(dataview.timestep), err_msg="Predicate selection changed timestep context!")
                    expected_ids = [[part.id for part in snap if part.type == predicate_type] for snap in self.part_snapshots[object_name]]
                    np.testing.assert_allclose(selection.id.squeeze(axis=-1), expected_ids)

    def test_object_relations(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                particle_group = group_type.__name__
                dataview = H5DataSelector(h5_file, particle_group=particle_group)
                parents = sim_inst._collect_instances_recursively(
                    [obj for obj in sim_inst.objects if isinstance(obj, group_type)]
                    )

                for parent in parents:
                    if not getattr(parent, "associated_objects", None):
                        continue
                    parent_key = parent.__class__.__name__
                    child_key = parent.associated_objects[0].__class__.__name__
                    child_ids = [child.who_am_i for child in parent.associated_objects]
                    np.testing.assert_array_equal(
                        dataview.get_child_ids(parent_key, child_key, parent.who_am_i),
                        child_ids,
                        err_msg=f"{parent_key}_to_{child_key} child IDs do not match for parent {parent.who_am_i}!",
                    )
                    for child in parent.associated_objects:
                        expected_parent_ids = [
                            obj.who_am_i
                            for obj in sim_inst.objects
                            if child in (getattr(obj, "associated_objects", None) or [])
                        ]
                        np.testing.assert_array_equal(
                            dataview.get_parent_ids(parent_key, child_key, child.who_am_i),
                            expected_parent_ids,
                            err_msg=f"{parent_key}_to_{child_key} parent IDs do not match for child {child.who_am_i}!",
                        )

    def test_obsolete_IO(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                path_to_dump, GLOBAL_COUNTER = sim_inst.init_pickle_dump(
                    path_to_dump=os.path.join(tmpdirname, "testfile.p.gz"))
                dungeon_witch_list = list(sim_inst.sys.part.all())
                sim_inst.dump_to_init(path_to_dump, dungeon_witch_list, GLOBAL_COUNTER)
                path_to_dump, GLOBAL_COUNTER = sim_inst.load_pickle_dump(
                    os.path.join(tmpdirname, "testfile.p.gz"))
            except MissingFeature as excp:
                logging.warning(f"Skipping depreciated IO pipeline tests because it requires a feature that is not available.  Caught exception {excp}")

class ElastomerFixture(CommonH5DataSelectorTests, BaseTestCase):
    box_dim = [5,5,20]
    layer_height = 4
    n_part = 20
    observable_name = "magnetic_dipole_moment"

    @classmethod
    def build_fixture(cls):
        sim_inst.sys.box_l = cls.box_dim
        conf_point_dipole = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)
        point_dipoles = [PointDipolePermanent(config=conf_point_dipole) for _ in range(cls.n_part)]
        config_E = Elastomer.config.specify(
            layer_height=cls.layer_height, n_parts=cls.n_part, associated_objects=point_dipoles, espresso_handle=sim_inst.sys, seed=sim_inst.seed)
        elastomer=Elastomer(config=config_E)
        sim_inst.store_objects([elastomer])
        sim_inst.set_objects([elastomer])
        cls.group_types = [Elastomer, PointDipolePermanent]

class FilamentFixture(CommonH5DataSelectorTests, BaseTestCase):

    N_avog = 6.02214076e23
    sigma = 1.
    rho_si = 0.6*N_avog
    no_obj=30
    N = no_obj/3
    vol = N/rho_si
    box_l = pow(vol, 1/3)
    _box_l = box_l/0.4e-09
    box_dim = _box_l*np.ones(3)
    _rho = N/pow(_box_l, 3)

    sheets_per_quad = 3
    part_per_filament = 2
    no_crowders=10
    part_per_ligand=2

    @classmethod
    def build_fixture(cls):
        quartet_configuration = Quartet.config.specify(espresso_handle=sim_inst.sys)
        quartets = [Quartet(config=quartet_configuration) for _ in range(cls.no_obj)]
        sim_inst.store_objects(quartets)

        bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
        grouped_quartets = [quartets[i:i+cls.sheets_per_quad]
                            for i in range(0, len(quartets), cls.sheets_per_quad)]
        quadriplex_configuration_list = [
            Quadriplex.config.specify(size=6., espresso_handle=sim_inst.sys, bond_handle=bond_quad, associated_objects=elem)
            for elem in grouped_quartets
        ]

        quadriplexes = [Quadriplex(config=configuration) for configuration in quadriplex_configuration_list]
        sim_inst.store_objects(quadriplexes)
        bond_pass = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
        grouped_quadriplexes = [quadriplexes[i:i+cls.part_per_filament:]
                                for i in range(0, len(quadriplexes), cls.part_per_filament)]
        filament_configuration_list = [
            Filament.config.specify(sigma=6, size=6*cls.part_per_filament, n_parts=cls.part_per_filament, espresso_handle=sim_inst.sys, bond_handle=bond_pass, associated_objects=elem)
            for elem in grouped_quadriplexes
        ]
        filaments = [Filament(config=configuration) for configuration in filament_configuration_list]
        sim_inst.store_objects(filaments)
        sim_inst.set_objects(filaments)

        crowder_configuration=Crowder.config.specify(sigma=1., size=1., espresso_handle=sim_inst.sys)
        crowders = [Crowder(config=crowder_configuration) for _ in range(cls.no_crowders)]
        sim_inst.store_objects(crowders)
        sim_inst.set_objects(crowders)

        cls.group_types = [Filament, Crowder]

    def test_crowder_missing_relation_smoke(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            crowder = next(obj for obj in sim_inst.objects if isinstance(obj, Crowder))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                missing_children = H5DataSelector(
                    h5_file,
                    particle_group="Crowder",
                ).get_child_ids(
                    "Crowder",
                    "Quadriplex",
                    crowder.who_am_i,
                )
            self.assertIsNone(missing_children)
            self.assertGreaterEqual(len(caught), 1)

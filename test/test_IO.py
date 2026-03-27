import numpy as np
import espressomd
from create_system import sim_inst , BaseTestCase
from pressomancy.simulation import Filament, Quartet, Quadriplex, Crowder
from pressomancy.helper_functions import BondWrapper
from pressomancy.analysis import H5DataSelector, H5ObservableSelector
import h5py
import tempfile
import os
from pressomancy.helper_functions import MissingFeature
import logging

class IOTest(BaseTestCase):

    N_avog = 6.02214076e23
    sigma = 1.
    rho_si = 0.6*N_avog
    no_obj=30
    N = int(no_obj/3)
    vol = N/rho_si
    box_l = pow(vol, 1/3)
    _box_l = box_l/0.4e-09
    box_dim = _box_l*np.ones(3)
    _rho = N/pow(_box_l, 3)

    sheets_per_quad = 3
    part_per_filament = 2
    no_crowders=10
    part_per_ligand=2


    def setUp(self) -> None:
        sim_inst.set_sys()
        quartet_configuration = Quartet.config.specify(espresso_handle=sim_inst.sys)
        quartets = [Quartet(config=quartet_configuration) for x in range(self.no_obj)]
        sim_inst.store_objects(quartets)

        bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
        grouped_quartets = [quartets[i:i+self.sheets_per_quad]
                            for i in range(0, len(quartets), self.sheets_per_quad)]
        quadriplex_configuration_list = [Quadriplex.config.specify(size=6., espresso_handle=sim_inst.sys, bond_handle=bond_quad, associated_objects=elem) for elem in grouped_quartets]

        quadriplex = [Quadriplex(config=configuration) for configuration in quadriplex_configuration_list]
        sim_inst.store_objects(quadriplex)

        bond_pass = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
        grouped_quadriplexes = [quadriplex[i:i+self.part_per_filament:]
                                for i in range(0, len(quadriplex), self.part_per_filament)]
        filament_configuration_list = [Filament.config.specify(sigma=6,size=6*self.part_per_filament, n_parts=self.part_per_filament, espresso_handle=sim_inst.sys, bond_handle=bond_pass, associated_objects=elem) for elem in grouped_quadriplexes]
        self.filaments = [Filament(config=configuration) for configuration in filament_configuration_list]
        sim_inst.store_objects(self.filaments)
        sim_inst.set_objects(self.filaments)
        crowder_configuration=Crowder.config.specify(sigma=1., size=1., espresso_handle=sim_inst.sys)
        self.crowders = [Crowder(config=crowder_configuration)
                    for x in range(self.no_crowders)]
        sim_inst.store_objects(self.crowders)
        sim_inst.set_objects(self.crowders)
        
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    @staticmethod
    def basic_structure(data,step, part_no):
        np.testing.assert_equal(len(data.timestep), step+1, err_msg="Timestep length does not match!")
        np.testing.assert_equal(len(data.particles), part_no, err_msg="Particle count does not match!")
        times=[x for x in data.timestep]
        parts=[x for x in data.particles]
        np.testing.assert_equal(len(times), step+1, err_msg="Timestep length does not match!")
        np.testing.assert_equal(len(parts), part_no, err_msg="Particle count does not match!")
    
    @staticmethod
    def slicing_check(data):
        lens=[]
        all_ts = [x for x in data.timestep]
        slice_sel = data.timestep[0:2]
        lens.append(len(slice_sel.timestep))
        all_ts2 = [x for x in slice_sel.timestep]
        list_sel = data.timestep[[0, 1]]  
        lens.append(len(list_sel.timestep))        
        all_ts3 = [x for x in list_sel.timestep]
        tuple_sel = data.timestep[(0, 1)]    
        lens.append(len(tuple_sel.timestep))
        all_ts4 = [x for x in tuple_sel.timestep]
        int_sel = data.timestep[-1]     
        lens.append(len(int_sel.timestep))             
        all_ts5 = [x for x in int_sel.timestep]
        all_ts = [x for x in data.particles]
        slice_sel = data.particles[0:2]   
        lens.append(len(slice_sel.particles))           
        all_ts2 = [x for x in slice_sel.particles]
        list_sel = data.particles[[0, 1]]    
        lens.append(len(list_sel.particles))      
        all_ts3 = [x for x in list_sel.particles]
        tuple_sel = data.particles[(0, 1)] 
        lens.append(len(tuple_sel.particles))
        all_ts4 = [x for x in tuple_sel.particles]
        int_sel = data.particles[-1]  
        lens.append(len(int_sel.particles))            
        all_ts5 = [x for x in int_sel.particles]
        np.testing.assert_array_equal(lens, [2, 2, 2, 1, 2, 2, 2 ,1], err_msg="Slicing did not return expected lengths!")

    @staticmethod
    def assert_step_time(h5_file, group_name, prop, expected_step, expected_time):
        step_value = h5_file[f"particles/{group_name}/{prop}/step"][-1]
        time_dataset = h5_file[f"particles/{group_name}/{prop}/time"]
        time_value = time_dataset[-1]
        np.testing.assert_equal(step_value, expected_step, err_msg=f"Stored step for {group_name}/{prop} does not match!")
        np.testing.assert_allclose(time_value, expected_time, err_msg=f"Stored time for {group_name}/{prop} does not match!")

    @staticmethod
    def assert_box_metadata(h5_file, group_name, expected_edges, expected_boundary=("periodic", "periodic", "periodic")):
        box_group = h5_file[f"particles/{group_name}/box"]
        expected_edges = np.array(expected_edges, dtype=float, copy=True)
        np.testing.assert_equal(int(box_group.attrs["dimension"]), len(expected_edges), err_msg=f"Box dimension for {group_name} does not match!")
        boundary = tuple(
            item.decode("ascii") if isinstance(item, bytes) else str(item)
            for item in np.atleast_1d(box_group.attrs["boundary"]).tolist()
        )
        np.testing.assert_equal(boundary, expected_boundary, err_msg=f"Box boundary for {group_name} does not match!")
        np.testing.assert_allclose(box_group["edges"][:], expected_edges, err_msg=f"Box edges for {group_name} do not match!")

    @staticmethod
    def assert_box_reader(data, expected_edges, expected_boundary=("periodic", "periodic", "periodic")):
        box = data.get_box()
        expected_edges = np.array(expected_edges, dtype=float, copy=True)
        np.testing.assert_equal(box["dimension"], len(expected_edges), err_msg="Box dimension from selector does not match!")
        np.testing.assert_equal(box["boundary"], expected_boundary, err_msg="Box boundary from selector does not match!")
        np.testing.assert_allclose(box["edges"], expected_edges, err_msg="Box edges from selector do not match!")

    @staticmethod
    def get_and_check(data, view_type, identity, ref_parts):
        properties=['pos','f','dip']
        for prop in properties:
            property_data_h5df=getattr(data.select_particles_by_object(object_name=view_type,connectivity_value=identity).timestep[-1],prop)
            property_data=[getattr(part,prop) for part in ref_parts]
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'
        for prop in ['id','type']:
            property_data_h5df=getattr(data.select_particles_by_object(object_name=view_type,connectivity_value=identity).timestep[-1],prop).flatten()
            property_data=[getattr(part,prop) for part in ref_parts]
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'
    
    @staticmethod
    def poke_analysis_api(data, view_type, identity, control_ids, ref_parts):
        filament_ids=np.array(list(data.get_connectivity_values(view_type)),dtype=int)
        np.testing.assert_array_equal(control_ids, filament_ids, err_msg=f"{view_type} IDs do not match!")
        filam = data.select_particles_by_object(object_name=view_type,connectivity_value=identity)
        property_data=[getattr(part,"id") for part in ref_parts]
        np.testing.assert_array_equal(filam.timestep[-1].id.flatten(), property_data, err_msg="Particle IDs do not match!")
            
    @staticmethod
    def get_child_poke(data, parent_ids, child_ids):
        quad_ids=[]
        for ide in parent_ids:
            quad_ids.extend(data.get_child_ids("Filament", "Quadriplex", parent_id=ide)) 
        np.testing.assert_array_equal(child_ids, quad_ids, err_msg="Quadiplex IDs do not match!")

    @staticmethod
    def get_parent_poke(data, parent_ids, child_ids):
        filam_ids=[]
        for ide in child_ids:
            filam_ids.extend(data.get_parent_ids("Filament", "Quadriplex", child_id=ide)) 
        # there are two childs per parent here so we subsample by 2
        np.testing.assert_array_equal(parent_ids, [filam_ids[i] for i in range(0,len(filam_ids),IOTest.part_per_ligand)], err_msg="Parent IDs do not match!")

    @staticmethod
    def cover_selector_exception_paths(data):
        """Trigger selector misuse branches for coverage without asserting strict failure."""
        tests = [
            # (callable, expected exception)
            (lambda: H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="DangerNoodle"), ValueError),
            (lambda: data.particles[-1], TypeError),
            (lambda: data[-1], TypeError),
            (lambda: iter(data), TypeError),
            (lambda: len(data), TypeError),
            (lambda: (
                H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Crowder")
                    .get_child_ids("Crowder", "Quadriplex", parent_id=0)
            ), KeyError),
        ]

        for fn, exc in tests:
            try:
                fn()
            except exc as e:
                print(f"{exc.__name__}: {e}")

    def test_IO_h5md(self):
        filam_ids=[filam.who_am_i for filam in self.filaments]
        quadriplex=[filam.associated_objects for filam in self.filaments]
        quadriplex_ids=[]
        for quads in quadriplex:
            for el in quads:
                quadriplex_ids.append(el.who_am_i)
                
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Build a temporary filename inside the directory
            h5_filename = os.path.join(tmpdirname, "testfile.h5")
            GLOBAL_COUNTER=sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename)
            for iid in range(2):
                frame_step = GLOBAL_COUNTER + 10 * (iid + 1)
                sim_inst.sys.integrator.run(1)
                sim_inst.write_part_group_to_h5(time_step=frame_step)
                data = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Filament")
                data_crowder = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Crowder")
                parts,_=self.filaments[iid].get_owned_part()
                self.basic_structure(data,iid,750)
                self.get_and_check(data, "Filament", iid, parts)
                self.poke_analysis_api(data, "Filament", iid, filam_ids, parts)
                self.get_child_poke(data, filam_ids, quadriplex_ids)
                self.get_parent_poke(data, filam_ids, quadriplex_ids)
                parts,_=self.crowders[iid].get_owned_part()
                self.get_and_check(data_crowder, "Crowder", iid, parts)
                self.basic_structure(data_crowder,iid,10)
                self.poke_analysis_api(data_crowder, "Crowder", iid, quadriplex_ids, parts)
                self.assert_step_time(sim_inst.io_dict['h5_file'], "Filament", "id", frame_step, sim_inst.sys.time)
                self.assert_step_time(sim_inst.io_dict['h5_file'], "Crowder", "id", frame_step, sim_inst.sys.time)
                self.assert_box_metadata(sim_inst.io_dict['h5_file'], "Filament", sim_inst.sys.box_l)
                self.assert_box_metadata(sim_inst.io_dict['h5_file'], "Crowder", sim_inst.sys.box_l)
                self.assert_box_reader(data, sim_inst.sys.box_l)
                self.assert_box_reader(data_crowder, sim_inst.sys.box_l)
            filament_time = sim_inst.io_dict['h5_file']["particles/Filament/id/time"][:]
            filament_step = sim_inst.io_dict['h5_file']["particles/Filament/id/step"][:]
            np.testing.assert_array_equal(filament_step, [10, 20], err_msg="Stored frame counters are incorrect!")
            self.assertTrue(np.all(np.diff(filament_time) > 0.0), "Stored physical times must increase monotonically.")
            self.assertFalse(np.allclose(filament_step.astype(float), filament_time), "Stored time should no longer mirror stored step.")
            data = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Filament")
            self.cover_selector_exception_paths(data)
            self.slicing_check(data)
            GLOBAL_COUNTER=sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename,mode='LOAD_NEW')
            self.assertEqual(GLOBAL_COUNTER, 2)
            data = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Filament")
            data_crowder = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Crowder")
            self.assert_box_reader(data, sim_inst.sys.box_l)
            self.assert_box_reader(data_crowder, sim_inst.sys.box_l)
            GLOBAL_COUNTER=sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename,mode='LOAD')
            self.assertEqual(GLOBAL_COUNTER, 2)
            data = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Filament")
            data_crowder = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Crowder")
            self.assert_box_reader(data, sim_inst.sys.box_l)
            self.assert_box_reader(data_crowder, sim_inst.sys.box_l)

    def test_IO_h5md_legacy_files_without_box(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "legacy_no_box.h5")
            sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename)

            for frame_step in (10, 20):
                sim_inst.sys.integrator.run(1)
                sim_inst.write_part_group_to_h5(time_step=frame_step)

            sim_inst.io_dict['h5_file'].flush()
            sim_inst.io_dict['h5_file'].close()
            sim_inst.io_dict['h5_file'] = None

            with h5py.File(h5_filename, "a") as h5_file:
                del h5_file["particles/Filament/box"]
                del h5_file["particles/Crowder/box"]

            with h5py.File(h5_filename, "r") as h5_file:
                filament_selector = H5DataSelector(h5_file, particle_group="Filament")
                crowder_selector = H5DataSelector(h5_file, particle_group="Crowder")
                with self.assertRaises(KeyError):
                    filament_selector.get_box()
                with self.assertRaises(KeyError):
                    crowder_selector.get_box()

            sim_inst.io_dict['flat_part_view'].clear()
            GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
                group_type=[Filament, Crowder],
                h5_data_path=h5_filename,
                mode='LOAD_NEW',
            )
            self.assertEqual(GLOBAL_COUNTER, 2)

            sim_inst.sys.integrator.run(1)
            sim_inst.write_part_group_to_h5(time_step=30)
            sim_inst.io_dict['h5_file'].flush()

            with h5py.File(h5_filename, "r") as h5_file:
                np.testing.assert_array_equal(
                    h5_file["particles/Filament/id/step"][:],
                    [10, 20, 30],
                    err_msg="Legacy file append after LOAD_NEW did not preserve step history.",
                )
                np.testing.assert_array_equal(
                    h5_file["particles/Crowder/id/step"][:],
                    [10, 20, 30],
                    err_msg="Legacy crowder file append after LOAD_NEW did not preserve step history.",
                )

            sim_inst.io_dict['flat_part_view'].clear()
            GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
                group_type=[Filament, Crowder],
                h5_data_path=h5_filename,
                mode='LOAD',
            )
            self.assertEqual(GLOBAL_COUNTER, 3)

    def test_mk_src_file_keeps_selected_frame_and_box_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            src_filename = os.path.join(tmpdirname, "src.h5")
            dst_filename = os.path.join(tmpdirname, "dst.h5")
            sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=src_filename)

            written_steps = []
            written_times = []
            for frame_step in (7, 13):
                sim_inst.write_part_group_to_h5(time_step=frame_step)
                written_steps.append(frame_step)
                written_times.append(sim_inst.sys.time)
                sim_inst.sys.integrator.run(1)

            sim_inst.mk_src_file(src_filename, dst_filename, prop_dim=[("v", 3)], time_step=1)

            with h5py.File(dst_filename, "r") as h5_file:
                pos_step = h5_file["particles/Filament/pos/step"][:]
                pos_time = h5_file["particles/Filament/pos/time"][:]
                v_step = h5_file["particles/Filament/v/step"][:]
                v_time = h5_file["particles/Filament/v/time"][:]

                np.testing.assert_array_equal(pos_step, [written_steps[1]], err_msg="mk_src_file did not keep the selected frame step.")
                np.testing.assert_allclose(pos_time, [written_times[1]], err_msg="mk_src_file did not keep the selected frame time.")
                np.testing.assert_array_equal(v_step, [written_steps[1]], err_msg="New property dataset did not inherit the kept frame step.")
                np.testing.assert_allclose(v_time, [written_times[1]], err_msg="New property dataset did not inherit the kept frame time.")
                self.assert_box_metadata(h5_file, "Filament", sim_inst.sys.box_l)

                selector = H5DataSelector(h5_file, particle_group="Filament")
                np.testing.assert_equal(len(selector.timestep), 1, err_msg="mk_src_file output should contain exactly one kept frame.")
                self.assert_box_reader(selector, sim_inst.sys.box_l)

    def test_write_registered_to_h5_keeps_streams_in_sync(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "mixed_streams.h5")
            magnetic_dipole_moment = np.zeros(3, dtype=np.float64)
            sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=h5_filename)
            sim_inst.inscribe_observable_group_to_h5(
                observable_defs=[("magnetic_dipole_moment", 3, np.float64, magnetic_dipole_moment)],
                h5_data_path=h5_filename,
                mode='NEW',
            )

            written_steps = []
            written_times = []
            written_values = []
            for frame_step, value in ((3, np.array([1.0, 2.0, 3.0])), (7, np.array([4.0, 5.0, 6.0]))):
                sim_inst.sys.integrator.run(1)
                magnetic_dipole_moment[:] = value
                sim_inst.write_registered_to_h5(time_step=frame_step)
                written_steps.append(frame_step)
                written_times.append(sim_inst.sys.time)
                written_values.append(value.copy())

            with h5py.File(h5_filename, "r") as h5_file:
                particle_step = h5_file["particles/Filament/id/step"][:]
                particle_time = h5_file["particles/Filament/id/time"][:]
                obs_group = h5_file["observables/magnetic_dipole_moment"]
                np.testing.assert_array_equal(particle_step, written_steps, err_msg="Particle frames do not match the unified write steps.")
                np.testing.assert_array_equal(obs_group["step"][:], written_steps, err_msg="Observable frames do not match the unified write steps.")
                np.testing.assert_allclose(particle_time, written_times, err_msg="Particle times do not match the unified write times.")
                np.testing.assert_allclose(obs_group["time"][:], written_times, err_msg="Observable times do not match the unified write times.")
                np.testing.assert_allclose(obs_group["value"][:], np.array(written_values), err_msg="Observable values do not match unified writes.")

            magnetic_dipole_moment[:] = np.array([7.0, 8.0, 9.0])
            sim_inst.write_observable_group_to_h5(time_step=11)
            sim_inst.io_dict['h5_file'].flush()
            sim_inst.io_dict['h5_file'].close()
            sim_inst.io_dict['h5_file'] = None
            sim_inst.io_dict['flat_part_view'].clear()

            particle_counter = sim_inst.inscribe_part_group_to_h5(
                group_type=[Filament],
                h5_data_path=h5_filename,
                mode='LOAD_NEW',
                force_resize_to_size=2,
            )
            observable_counter = sim_inst.inscribe_observable_group_to_h5(
                observable_defs=[("magnetic_dipole_moment", 3, np.float64, magnetic_dipole_moment)],
                h5_data_path=h5_filename,
                mode='LOAD_NEW',
                force_resize_to_size=2,
            )
            self.assertEqual(particle_counter, 2)
            self.assertEqual(observable_counter, 2)

            with h5py.File(h5_filename, "r") as h5_file:
                np.testing.assert_array_equal(h5_file["particles/Filament/id/step"][:], written_steps, err_msg="Particle stream should remain at the checkpointed counter.")
                np.testing.assert_array_equal(h5_file["observables/magnetic_dipole_moment/step"][:], written_steps, err_msg="Observable stream should be truncated back to the checkpointed counter.")

    def test_observables_h5md(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "testfile.h5")
            magnetic_dipole_moment = np.zeros(3, dtype=np.float64)
            sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=h5_filename)
            observable_counter = sim_inst.inscribe_observable_group_to_h5(
                observable_defs=[("magnetic_dipole_moment", 3, np.float64, magnetic_dipole_moment)],
                h5_data_path=h5_filename,
                mode='NEW',
            )
            self.assertEqual(observable_counter, 0)

            written_steps = []
            written_times = []
            written_values = []
            for frame_step, value in ((3, np.array([1.0, 2.0, 3.0])), (7, np.array([4.0, 5.0, 6.0]))):
                sim_inst.sys.integrator.run(1)
                magnetic_dipole_moment[:] = value
                sim_inst.write_observable_group_to_h5(time_step=frame_step)
                written_steps.append(frame_step)
                written_times.append(sim_inst.sys.time)
                written_values.append(value.copy())

            with h5py.File(h5_filename, "r") as h5_file:
                obs_group = h5_file["observables/magnetic_dipole_moment"]
                np.testing.assert_array_equal(obs_group["step"][:], written_steps, err_msg="Observable steps do not match appended data.")
                np.testing.assert_allclose(obs_group["time"][:], written_times, err_msg="Observable times do not match appended data.")
                np.testing.assert_allclose(obs_group["value"][:], np.array(written_values), err_msg="Observable values do not match appended data.")

            observable_counter = sim_inst.inscribe_observable_group_to_h5(
                observable_defs=[("magnetic_dipole_moment", 3, np.float64, magnetic_dipole_moment)],
                h5_data_path=h5_filename,
                mode='LOAD_NEW',
            )
            self.assertEqual(observable_counter, 2)

            observable_counter = sim_inst.inscribe_observable_group_to_h5(
                observable_defs=[("magnetic_dipole_moment", 3, np.float64, magnetic_dipole_moment)],
                h5_data_path=h5_filename,
                mode='LOAD_NEW',
                force_resize_to_size=1,
            )
            self.assertEqual(observable_counter, 1)

            with h5py.File(h5_filename, "r") as h5_file:
                obs_group = h5_file["observables/magnetic_dipole_moment"]
                np.testing.assert_array_equal(obs_group["step"][:], [written_steps[0]], err_msg="Observable resize did not preserve the leading frame step.")
                np.testing.assert_allclose(obs_group["time"][:], [written_times[0]], err_msg="Observable resize did not preserve the leading frame time.")
                np.testing.assert_allclose(obs_group["value"][:], np.array([written_values[0]]), err_msg="Observable resize did not preserve the leading frame value.")

    def test_observable_selector_h5md(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "observable_selector.h5")
            magnetic_dipole_moment = np.zeros(3, dtype=np.float64)
            observable_counter = sim_inst.inscribe_observable_group_to_h5(
                observable_defs=[("magnetic_dipole_moment", 3, np.float64, magnetic_dipole_moment)],
                h5_data_path=h5_filename,
                mode='NEW',
            )
            self.assertEqual(observable_counter, 0)

            written_steps = []
            written_times = []
            written_values = []
            for frame_step, value in ((3, np.array([1.0, 2.0, 3.0])), (7, np.array([4.0, 5.0, 6.0]))):
                sim_inst.sys.integrator.run(1)
                magnetic_dipole_moment[:] = value
                sim_inst.write_observable_group_to_h5(time_step=frame_step)
                written_steps.append(frame_step)
                written_times.append(sim_inst.sys.time)
                written_values.append(value.copy())

            with h5py.File(h5_filename, "r") as h5_file:
                selector = H5ObservableSelector(h5_file, observable_name="magnetic_dipole_moment")
                np.testing.assert_array_equal(selector.step, written_steps, err_msg="Observable selector did not return the stored step values.")
                np.testing.assert_allclose(selector.time, written_times, err_msg="Observable selector did not return the stored times.")
                np.testing.assert_allclose(selector.value, np.array(written_values), err_msg="Observable selector did not return the stored values.")
                sliced = selector.timestep[0:2]
                np.testing.assert_equal(len(sliced.timestep), 2, err_msg="Observable timestep slicing did not preserve the expected frame count.")
                np.testing.assert_array_equal(sliced.step, written_steps, err_msg="Observable timestep slicing did not preserve the expected steps.")
                frames = [frame for frame in selector.timestep]
                np.testing.assert_equal(len(frames), 2, err_msg="Observable timestep iteration did not yield the expected number of frame selectors.")
                np.testing.assert_array_equal(frames[0].step, written_steps[0], err_msg="Observable timestep iteration did not keep the expected stored step for the first frame.")
                np.testing.assert_allclose(frames[0].time, written_times[0], err_msg="Observable timestep iteration did not keep the expected stored time for the first frame.")
                np.testing.assert_allclose(frames[0].value, written_values[0], err_msg="Observable timestep iteration did not keep the expected stored value for the first frame.")
                with self.assertRaises(TypeError):
                    selector[0]

            with h5py.File(h5_filename, "r") as h5_file:
                with self.assertRaises(ValueError):
                    H5ObservableSelector(h5_file, observable_name="missing_observable")

            with h5py.File(h5_filename, "a") as h5_file:
                h5_file["observables/magnetic_dipole_moment/step"].resize((1,))
                with self.assertRaises(ValueError):
                    H5ObservableSelector(h5_file, observable_name="magnetic_dipole_moment")

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
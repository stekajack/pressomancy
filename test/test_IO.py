import numpy as np
import espressomd
from create_system import sim_inst , BaseTestCase
from pressomancy.simulation import Filament, Quartet, Quadriplex, Crowder
from pressomancy.helper_functions import BondWrapper
from pressomancy.analysis import H5DataSelector
import h5py
import tempfile
import os
from pathlib import Path
from pressomancy.helper_functions import MissingFeature
import logging
from unittest.mock import patch
import pressomancy.simulation as simulation_module

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
        self.cleanup()
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
    def exceptions(data):
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
                sim_inst.sys.integrator.run(1)
                sim_inst.write_part_group_to_h5(time_step=GLOBAL_COUNTER)
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
            data = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Filament")
            predicate_cutoff = max(part.id for part in self.filaments[1].get_owned_part()[0])
            predicate = lambda subset: subset.timestep[-1].id.flatten()[0] <= predicate_cutoff
            np.testing.assert_array_equal(data.get_connectivity_values("Filament", predicate=predicate), filam_ids[:2], err_msg="Predicate-filtered filament IDs do not match!")
            np.testing.assert_array_equal(data.get_connectivity_values("Filament", predicate=predicate, fast=True), filam_ids[:2], err_msg="Fast predicate-filtered filament IDs do not match!")
            self.exceptions(data)
            self.slicing_check(data)
            GLOBAL_COUNTER=sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename,mode='LOAD_NEW')
            GLOBAL_COUNTER=sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename,mode='LOAD')

    def test_h5md_optional_metadata_and_load_new_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "testfile.h5")
            with patch.object(simulation_module, "get_submission_creator_info", return_value=("UPLOAD_VIEW/suspensions/mcp_suspension_pressomancy.py", "main@abc1234-dirty")), patch.object(simulation_module, "get_repo_context", return_value=(Path("/home/stekajack/pressomancy"), "main@def5678-dirty")):
                GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename)
            sim_inst.sys.integrator.run(1)
            sim_inst.write_part_group_to_h5(time_step=GLOBAL_COUNTER)

            h5_handle = sim_inst.io_dict['h5_file']
            self.assertIn("h5md", h5_handle)
            np.testing.assert_array_equal(h5_handle["h5md"].attrs["version"], np.array([1, 0], dtype=np.int32))
            self.assertEqual(h5_handle["h5md/author"].attrs["name"], "unknown")
            self.assertEqual(h5_handle["h5md/creator"].attrs["name"], "UPLOAD_VIEW/suspensions/mcp_suspension_pressomancy.py")
            self.assertEqual(h5_handle["h5md/creator"].attrs["version"], "main@abc1234-dirty")
            self.assertEqual(h5_handle["parameters/pressomancy"].attrs["version"], "main@def5678-dirty")
            self.assertEqual(h5_handle["parameters/pressomancy/part_types"].attrs["real"], 1)

            data = H5DataSelector(h5_handle, particle_group="Filament")
            np.testing.assert_array_equal(data.metadata["h5md"]["_meta"]["attributes"]["version"], np.array([1, 0], dtype=np.int32))
            self.assertEqual(data.metadata["h5md"]["creator"]["_meta"]["attributes"]["name"], "UPLOAD_VIEW/suspensions/mcp_suspension_pressomancy.py")
            self.assertEqual(data.metadata["parameters"]["pressomancy"]["_meta"]["attributes"]["version"], "main@def5678-dirty")
            self.assertEqual(data.metadata["parameters"]["pressomancy"]["part_types"]["_meta"]["attributes"]["real"], 1)

            original_part_types = {key: value for key, value in sim_inst.part_types.items() if isinstance(value, int)}
            sim_inst.part_types.clear()
            sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename,mode='LOAD_NEW')
            for key, value in original_part_types.items():
                self.assertEqual(dict.get(sim_inst.part_types, key), value)
            sim_inst.io_dict['h5_file'].close()

        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "testfile_legacy.h5")
            GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename)
            sim_inst.sys.integrator.run(1)
            sim_inst.write_part_group_to_h5(time_step=GLOBAL_COUNTER)
            h5_handle = sim_inst.io_dict['h5_file']
            del h5_handle["parameters/pressomancy/part_types"]
            h5_handle["particles/Filament/type/value"][-1, 0, 0] = 999
            h5_handle.flush()
            h5_handle.close()

            sim_inst.part_types.clear()
            with self.assertLogs(level="WARNING") as log_context:
                sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename, mode='LOAD_NEW')
            self.assertEqual(dict.get(sim_inst.part_types, "real"), 1)
            self.assertTrue(any("Unmatched numeric types=[999]" in entry for entry in log_context.output))
            sim_inst.io_dict['h5_file'].close()

        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "testfile_author.h5")
            sim_inst.set_author("Alice", "alice@example.com")
            with patch.object(simulation_module, "get_submission_creator_info", return_value=("unknown/mcp_suspension_pressomancy.py", "unknown")), patch.object(simulation_module, "get_repo_context", return_value=(None, "unknown")):
                sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename)
            h5_handle = sim_inst.io_dict['h5_file']
            self.assertEqual(h5_handle["h5md/author"].attrs["name"], "Alice")
            self.assertEqual(h5_handle["h5md/author"].attrs["email"], "alice@example.com")
            self.assertEqual(h5_handle["h5md/creator"].attrs["name"], "unknown/mcp_suspension_pressomancy.py")
            self.assertEqual(h5_handle["parameters/pressomancy"].attrs["version"], "unknown")
            h5_handle.close()

        with tempfile.TemporaryDirectory() as tmpdirname:
            h5_filename = os.path.join(tmpdirname, "testfile_src.h5")
            dest_filename = os.path.join(tmpdirname, "testfile_src_shrunk.h5")
            with patch.object(simulation_module, "get_submission_creator_info", return_value=("UPLOAD_VIEW/suspensions/mcp_suspension_pressomancy.py", "main@abc1234-dirty")), patch.object(simulation_module, "get_repo_context", return_value=(Path("/home/stekajack/pressomancy"), "main@def5678-dirty")):
                GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(group_type=[Filament, Crowder], h5_data_path=h5_filename)
            sim_inst.sys.integrator.run(1)
            sim_inst.write_part_group_to_h5(time_step=GLOBAL_COUNTER)
            sim_inst.io_dict['h5_file'].flush()
            sim_inst.io_dict['h5_file'].close()
            sim_inst.mk_src_file(h5_filename, dest_filename, time_step=0)

            with h5py.File(dest_filename, "r") as h5_file:
                self.assertEqual(h5_file["h5md/creator"].attrs["name"], "UPLOAD_VIEW/suspensions/mcp_suspension_pressomancy.py")
                self.assertEqual(h5_file["h5md/creator"].attrs["version"], "main@abc1234-dirty")
                self.assertEqual(h5_file["parameters/pressomancy"].attrs["version"], "main@def5678-dirty")
                self.assertEqual(h5_file["parameters/pressomancy/part_types"].attrs["real"], 1)

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


   

        


        
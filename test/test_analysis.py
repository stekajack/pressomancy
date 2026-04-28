import numpy as np
from create_system import sim_inst , BaseTestCase
from pressomancy.simulation import Elastomer, PointDipolePermanent
from pressomancy.analysis import H5DataSelector
import tempfile
import os

class cestica():

    def __init__(self):
        pass
        

class H5DataSelectorTest(BaseTestCase):

    box_dim = [5,5,20]
    layer_height = 4
    n_part = 20

    def setUp(self) -> None:
        self.part_snapshots=[]
        sim_inst.sys.box_l = self.box_dim
        conf_point_dipole = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)
        point_dipoles = [PointDipolePermanent(config=conf_point_dipole) for _ in range(self.n_part)]
        config_E = Elastomer.config.specify(
            layer_height=self.layer_height, n_parts=self.n_part, associated_objects=point_dipoles, espresso_handle=sim_inst.sys, seed=sim_inst.seed)
        self.elastomer=Elastomer(config=config_E)
        sim_inst.store_objects([self.elastomer])
        sim_inst.set_objects([self.elastomer])
        self.elastomer.create_substrate(geometry="part")
                
    def tearDown(self) -> None:
        self.part_snapshots=[]
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part),0)

    
    def capture_particle_snapshot(self, parts):
        snap=[]
        for part in parts:
            new=cestica()
            for prop, _ in sim_inst.io_dict["properties"]:
                setattr(new, prop, getattr(part, prop))
            snap.append(new)
        return snap

    @staticmethod
    def get_and_check_complete_object(data, view_type, identity, ref_parts,predicate_type_list,time_slice):
        selection_source = data if time_slice is None else data.timestep[time_slice]
        selection=selection_source.select_particles_by_object(object_name=view_type, connectivity_value=identity)
        for prop,shape in sim_inst.io_dict['properties']:
            property_data_h5df=getattr(selection,prop)
            property_data=[]
            time_part_slice=np.atleast_2d(ref_parts if time_slice is None else ref_parts[time_slice])
            for snap in time_part_slice:
                property_data.append([getattr(part,prop) for part in snap])
            if shape == 1:
                property_data_h5df=np.squeeze(property_data_h5df, axis=-1)
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'
        for predicate_type in predicate_type_list:
            selection=selection_source.select_particles_by_object(object_name=view_type, connectivity_value=identity,predicate=lambda p:p.type==predicate_type)
            for prop,shape in sim_inst.io_dict['properties']:
                property_data_h5df=getattr(selection,prop)
                property_data=[]
                time_part_slice=np.atleast_2d(ref_parts if time_slice is None else ref_parts[time_slice])
                for snap in time_part_slice:
                    property_data.append([getattr(part,prop) for part in snap  if part.type==predicate_type])
                if shape == 1:
                    property_data_h5df=np.squeeze(property_data_h5df, axis=-1)
                assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'


    @staticmethod
    def get_and_check_connectivity_predicate(data, view_type, particle_type, control_ids):
        selected_ids = np.array(
            data.get_connectivity_values(
                view_type,
                predicate=lambda subset: np.all(subset.timestep[-1].type == particle_type),
            ),
            dtype=int,
        )
        np.testing.assert_array_equal(control_ids, selected_ids, err_msg=f"{view_type} predicate-filtered IDs do not match!")

    def test_select_particles_by_object(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Build a temporary filename inside the directory
            h5_filename = os.path.join(tmpdirname, "testfile.h5")
            GLOBAL_COUNTER=sim_inst.inscribe_part_group_to_h5(group_type=[Elastomer], h5_data_path=h5_filename)
            parts,_=self.elastomer.get_owned_part()
            pdp_parts = [obj.get_owned_part()[0][0] for obj in self.elastomer.associated_objects]
            pdp_snapshots=[]
            for _ in range(4):
                sim_inst.sys.integrator.run(1)
                sim_inst.write_part_group_to_h5(time_step=GLOBAL_COUNTER)
                self.part_snapshots.append(self.capture_particle_snapshot(parts))
                pdp_snapshots.append(self.capture_particle_snapshot(pdp_parts))
            dataview = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Elastomer")
            assert len(parts) == len(list(sim_inst.sys.part.all())), f"Expected all particles to be owned by the Elastomer, but found {len(parts)} out of {len(list(sim_inst.sys.part.all()))}!"
            for time_slice in [None, -1, 0, slice(0, 2, 1)]:
                self.get_and_check_complete_object(dataview, "Elastomer", 0, self.part_snapshots, [98, 61], time_slice=time_slice)
            pdp_ids = np.array([obj.who_am_i for obj in self.elastomer.associated_objects], dtype=int)
            ids = dataview.get_connectivity_values("PointDipolePermanent")
            np.testing.assert_array_equal(ids, pdp_ids, err_msg="PointDipolePermanent connectivity IDs do not match Espresso-owned object ids!")
            for time_slice in [None, -1, 0, slice(0, 2, 1)]:
                self.get_and_check_complete_object(dataview, "PointDipolePermanent", pdp_ids, pdp_snapshots, [61], time_slice=time_slice)
            selection = dataview.select_particles_by_object(
                object_name="PointDipolePermanent",
                connectivity_value=pdp_ids,
                predicate=lambda subset: subset.timestep[-1].type == sim_inst.part_types["pdp_real"],
            )
            np.testing.assert_equal(len(selection.timestep), len(dataview.timestep), err_msg="Predicate selection changed timestep context!")
            np.testing.assert_allclose(selection.id.squeeze(axis=-1), [[part.id for part in snap] for snap in pdp_snapshots])
            pdp_ids = np.array([obj.who_am_i for obj in self.elastomer.associated_objects], dtype=int)
            self.get_and_check_connectivity_predicate(dataview, "PointDipolePermanent", sim_inst.part_types["pdp_real"], pdp_ids)

import numpy as np
from create_system import sim_inst , BaseTestCase
from pressomancy.simulation import Elastomer, PointDipolePermanent
from pressomancy.analysis import H5DataSelector
import tempfile
import os
from unittest.mock import patch

class H5DataSelectorTest(BaseTestCase):

    box_dim = [10,10,40]
    layer_height = 4
    n_part = 20

    def setUp(self) -> None:
        sim_inst.set_sys()
        config_E = Elastomer.config.specify(layer_height=self.layer_height, n_parts=self.n_part,
            associated_objects=[PointDipolePermanent(config=PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)) for _ in range(self.n_part)],
                                           espresso_handle=sim_inst.sys, seed=sim_inst.seed)
        elastomer=[Elastomer(config=config_E) for _ in range(1)]
        sim_inst.store_objects(elastomer)
        sim_inst.set_objects(elastomer)
        self.elastomer= elastomer[0]
        self.elastomer.create_substrate(geometry="part")
                
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    @staticmethod
    def get_and_check_complete_object(data, view_type, ref_parts):
        properties=['pos','f','dip']
        for prop in properties:
            property_data_h5df=getattr(data.select_particles_by_object(object_name=view_type).timestep[-1],prop)
            property_data=[getattr(part,prop) for part in ref_parts]
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'
        for prop in ['id','type']:
            property_data_h5df=getattr(data.select_particles_by_object(object_name=view_type).timestep[-1],prop).flatten()
            property_data=[getattr(part,prop) for part in ref_parts]
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'
        for prop in ['id','type']:
            property_data_h5df=getattr(data.select_particles_by_object(object_name=view_type,predicate=lambda ds: getattr(ds, prop) == 98).timestep[-1],prop).flatten()
            property_data=[getattr(part,prop) for part in ref_parts if getattr(part,prop)==98]
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'
    
    @staticmethod
    def get_and_check_from_type(data, view_type, particle_types, ref_parts):
        for prop in ['id','type']:
            property_data_h5df=getattr(data.select_particles_by_object(object_name=view_type,predicate=lambda ds: np.isin(ds.type, particle_types)).timestep[-1],prop).flatten()
            property_data=[getattr(part,prop) for part in ref_parts]
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'

    def test_select_particles_by_object(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Build a temporary filename inside the directory
            h5_filename = os.path.join(tmpdirname, "testfile.h5")
            GLOBAL_COUNTER=sim_inst.inscribe_part_group_to_h5(group_type=[Elastomer], h5_data_path=h5_filename)
            for _ in range(2):
                sim_inst.sys.integrator.run(1)
                sim_inst.write_part_group_to_h5(time_step=GLOBAL_COUNTER)
                data = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group="Elastomer")
                parts,_=self.elastomer.get_owned_part()
                self.get_and_check_complete_object(data, "Elastomer", parts)
                parts = sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"])
                assert len(parts) == self.n_part
                self.get_and_check_complete_object(data, "PointDipolePermanent", parts)
                parts = sim_inst.sys.part.select(type=sim_inst.part_types["substrate"])
                self.get_and_check_from_type(data, "Elastomer", sim_inst.part_types["substrate"], parts)
import numpy as np
import espressomd
from create_system import sim_inst , BaseTestCase
from pressomancy.simulation import Filament, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
from pressomancy.analysis import H5DataSelector
import h5py
import tempfile
import os

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
        filaments = [Filament(config=configuration) for configuration in filament_configuration_list]
        sim_inst.store_objects(filaments)
        sim_inst.set_objects(filaments)
        
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    @staticmethod
    def get_and_check(data,time):
        data = H5DataSelector(data, particle_group="Filament")
        properties=['pos','f','dip']
        for prop in properties:
            property_data=getattr(sim_inst.sys.part.all(),prop)
            property_data_h5df=getattr(data.timestep[time].particles[:],prop)
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'
        for prop in ['id','type']:
            property_data=getattr(sim_inst.sys.part.all(),prop)
            property_data_h5df=getattr(data.timestep[time].particles[:],prop).flatten()
            assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), f'The vectors differ!, {property_data}, {property_data_h5df}'

    def test_IO_h5md(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Build a temporary filename inside the directory
            h5_filename = os.path.join(tmpdirname, "testfile.h5")
            with h5py.File(h5_filename, "w") as f:
                hot_potato=sim_inst.inscribe_part_group_to_h5(group_type=Filament, h5_file=f)
                for bookkeeper in range(10):
                    sim_inst.sys.integrator.run(1)
                    sim_inst.write_part_group_to_h5(config=hot_potato,time_step=0,h5_file=f)
                    self.get_and_check(f,bookkeeper)

        
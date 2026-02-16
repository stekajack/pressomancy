from pressomancy.simulation import PointDipolePermanent, PointDipoleSuperpara
from create_system import sim_inst, BaseTestCase
import numpy as np

class PointDipoleTest(BaseTestCase):
    H_ext = [0,0,3.]

    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part),0)

    def test_set_object_generic(self):
        mag_part = [PointDipolePermanent(config=PointDipolePermanent.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)
        mag_part =  [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)

    def test_set_object(self):
        mag_part = [PointDipolePermanent(
                    config=PointDipolePermanent.config.specify(dipm=0.7, size=2.,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)
        mag_part = [PointDipoleSuperpara(
                    config=PointDipoleSuperpara.config.specify(dipm=1., size=0.5,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)

    def test_magnetize_PointDipoleSuperpara(self):
        mag_part = [PointDipoleSuperpara(
                    config=PointDipoleSuperpara.config.specify(dipm=1., size=0.5,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)

        part_list = list(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]))

        sim_inst.magnetize(part_list, mag_part[0].params["dipm"], H_ext=self.H_ext)

    def test_magnetize_mixture(self):
        mag_part_pds = [PointDipoleSuperpara(
                    config=PointDipoleSuperpara.config.specify(dipm=1., size=0.5,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part_pds)
        sim_inst.set_objects(mag_part_pds)
        mag_part_pdp = [PointDipolePermanent(
                    config=PointDipolePermanent.config.specify(dipm=0.7, size=2.,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part_pdp)
        sim_inst.set_objects(mag_part_pdp)

        part_list = list(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]))

        sim_inst.magnetize(part_list, mag_part_pds[0].params["dipm"], H_ext=self.H_ext)

    def test_other_magnetize(self):
        mag_part_pds = [PointDipoleSuperpara(
                    config=PointDipoleSuperpara.config.specify(dipm=1., size=0.5,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part_pds)
        sim_inst.set_objects(mag_part_pds)
        mag_part_pdp = [PointDipolePermanent(
                    config=PointDipolePermanent.config.specify(dipm=0.7, size=2.,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part_pdp)
        sim_inst.set_objects(mag_part_pdp)

        part_list = list(sim_inst.sys.part.select(type=sim_inst.part_types["pds_real"]))

        sim_inst.magnetize_lin(part_list, mag_part_pds[0].params["dipm"], H_ext=self.H_ext, Xi=0.3)
        sim_inst.magnetize_froelich_kennelly(part_list, mag_part_pds[0].params["dipm"], H_ext=self.H_ext, Xi=0.5)
        sim_inst.magnetize_dumb(part_list, mag_part_pds[0].params["dipm"], H_ext=self.H_ext)
    
from pressomancy.simulation import PointDipolePermanent, PointDipoleSuperpara
from create_system import sim_inst, BaseTestCase

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
        assert sim_inst.part_types["pds_real"] == 62 and sim_inst.part_types["pds_virt"] == 666

    def test_set_object(self):
        mag_part = [PointDipolePermanent(
                    config=PointDipolePermanent.config.specify(dipm=1.2, size=2.,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)
        mag_part = [PointDipoleSuperpara(
                    config=PointDipoleSuperpara.config.specify(dipm=1., Xi_0=0.1, size=0.5,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)

    def test_magnetize_PointDipoleSuperpara(self):
        mag_part = [PointDipoleSuperpara(
                    config=PointDipoleSuperpara.config.specify(dipm=1., size=0.5,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part)
        sim_inst.set_objects(mag_part)

        sim_inst.time_step = 0.001
        sim_inst.sys.integrator.run(0, recalc_forces=True)

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

        sim_inst.time_step = 0.001
        sim_inst.sys.integrator.run(0, recalc_forces=True)

    def test_magnetize_Froelich_Kennelly(self):
        mag_part_pds = [PointDipoleSuperpara(
                    config=PointDipoleSuperpara.config.specify(dipm=1., mag_func=1, size=0.5,
                    espresso_handle=sim_inst.sys)) for _ in range(10)]
        sim_inst.store_objects(mag_part_pds)
        sim_inst.set_objects(mag_part_pds)
        mag_part_pdp = [PointDipolePermanent(
                    config=PointDipolePermanent.config.specify(dipm=0.7, size=2.,
                    espresso_handle=sim_inst.sys)) for _ in range(10)] # mag_func = 0 for langevin and 1 for Froelich-Kennelly
        sim_inst.store_objects(mag_part_pdp)
        sim_inst.set_objects(mag_part_pdp)

        sim_inst.time_step = 0.001
        sim_inst.sys.integrator.run(0, recalc_forces=True)
    
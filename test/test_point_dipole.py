from pressomancy.simulation import PointDipolePermanent, PointDipoleSuperpara
from pressomancy.helper_functions import api_agnostic_feature_check
from create_system import sim_inst, BaseTestCase


class PointDipoleTest(BaseTestCase):
    H_ext = [0,0,3.]
    config=PointDipolePermanent.config.specify(
        dipm=1.2, size=2., espresso_handle=sim_inst.sys)

    def tearDown(self) -> None:
        self.mag_part=None
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part),0)
    
    def setUp(self) -> None:
        self.mag_part = [PointDipolePermanent(config=PointDipolePermanent.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        self.mag_part.append(PointDipolePermanent(config=self.config))
        sim_inst.store_objects(self.mag_part)
        sim_inst.set_objects(self.mag_part)

    def test_set_object_generic(self):
        
        assert sim_inst.part_types["pdp_real"] == 61
    
if all(api_agnostic_feature_check(feature) for feature in PointDipoleSuperpara.required_features):
    class PointDipoleSuperparaTest(BaseTestCase):
        H_ext = [0,0,3.]
        config=PointDipoleSuperpara.config.specify(
            dipm=1., size=0.5, espresso_handle=sim_inst.sys)

        def tearDown(self) -> None:
            self.mag_part=None
            self.cleanup()
            self.assertEqual(len(sim_inst.sys.part),0)
        
        def setUp(self) -> None:
            self.mag_part = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
            self.mag_part.append(PointDipoleSuperpara(config=self.config))
            sim_inst.store_objects(self.mag_part)
            sim_inst.set_objects(self.mag_part)

        def test_set_object_generic(self):
            assert sim_inst.part_types["pds_real"] == 62 and sim_inst.part_types["pds_virt"] == 666
            p_virt = next(iter(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"])))
            assert p_virt.magnetodynamics.ideal["is_enabled"] is True
            assert p_virt.magnetodynamics.ideal["sat_mag"] == 1.0
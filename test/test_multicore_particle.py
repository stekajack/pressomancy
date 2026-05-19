import numpy as np
import espressomd
from create_system import sim_inst, BaseTestCase
from pressomancy.simulation import MulticorePart
from pressomancy.helper_functions import api_agnostic_feature_check
if espressomd.version.major() == 5:
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation

class MulticorePartTest(BaseTestCase):
    def tearDown(self) -> None:
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part), 0)

    def test_add_dipole_moments_to_virtuals_infinite(self):
        instance = MulticorePart(config=MulticorePart.config.specify(espresso_handle=sim_inst.sys))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        dip_moments = np.asarray([[1., 0., 0.], [0., 1., 0.]])
        instance.add_dipole_moments_to_virtuals(dip_moments=dip_moments,anisotropy={'kind': 'infinite', 'params': {}})

        virt_parts = instance.type_part_dict['virt']
        assigned_dips = [np.asarray(part.dip) for part in virt_parts if np.linalg.norm(part.dip) > 0]

        self.assertEqual(len(assigned_dips), len(dip_moments))
        for dip_moment in dip_moments:
            self.assertTrue(any(np.allclose(dip_moment, assigned_dip) for assigned_dip in assigned_dips))

    def test_add_dipole_moments_to_virtuals_finite_egg_missing_params(self):
        instance = MulticorePart(config=MulticorePart.config.specify(espresso_handle=sim_inst.sys))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])

        with self.assertRaisesRegex(ValueError, 'egg_gamma'):
            instance.add_dipole_moments_to_virtuals(
                dip_moments=np.asarray([[1., 0., 0.]]),
                anisotropy={'kind': 'finite_egg', 'params': {'aniso_energy': 2.0}})


if all(api_agnostic_feature_check(feature) for feature in ['EGG_MODEL']):
    class MulticorePartEggTest(BaseTestCase):
        def tearDown(self) -> None:
            self.cleanup()
            self.assertEqual(len(sim_inst.sys.part), 0)

        def test_add_dipole_moments_to_virtuals_finite_egg(self):
            sim_inst.sys.thermostat.set_brownian(kT=1.0, gamma=1.0, seed=17)
            sim_inst.sys.integrator.set_brownian_dynamics()

            instance = MulticorePart(config=MulticorePart.config.specify(espresso_handle=sim_inst.sys))
            sim_inst.store_objects([instance])
            sim_inst.set_objects([instance])

            dip_moments = np.asarray([[1., 0., 0.], [0., 1., 0.]])
            instance.add_dipole_moments_to_virtuals(
                dip_moments=dip_moments,
                anisotropy={'kind': 'finite_egg', 'params': {'egg_gamma': 1.5, 'aniso_energy': 2.5}})

            self.assertEqual(len(instance.type_part_dict['yolk']), len(dip_moments))
            self.assertEqual(len(instance.type_part_dict['virt']),
                             instance.params['n_parts'] - 1 - len(dip_moments))

            assigned_dips = [np.asarray(part.dip) for part in instance.type_part_dict['yolk']]
            for dip_moment in dip_moments:
                self.assertTrue(any(np.allclose(dip_moment, assigned_dip) for assigned_dip in assigned_dips))

            for yolk in instance.type_part_dict['yolk']:
                self.assertTrue(yolk.magnetodynamics.egg['is_enabled'])
                self.assertAlmostEqual(yolk.magnetodynamics.egg['gamma'], 1.5)
                self.assertAlmostEqual(yolk.magnetodynamics.egg['anisotropy_energy'], 2.5)
                if espressomd.version.major() == 5:
                    self.assertEqual(yolk.propagation,
                                     Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT)
                    self.assertEqual(tuple(yolk.rotation), (True, True, True))

            sim_inst.sys.integrator.run(0, recalc_forces=True)

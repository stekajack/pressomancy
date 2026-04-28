import numpy as np

from create_system import sim_inst, BaseTestCase
from pressomancy.simulation import MulticorePart


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

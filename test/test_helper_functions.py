import unittest
import numpy as np

from pressomancy.helper_functions import get_perpendicular


class HelperFunctionsTest(unittest.TestCase):
    def test_get_perpendicular_random_path_returns_unit_perpendicular(self):
        vec = np.array([0.3, -0.4, 0.5])
        unit_vec = vec / np.linalg.norm(vec)
        results = []
        for _ in range(8):
            perp = get_perpendicular(vec, phi=None)
            results.append(perp)
            self.assertTrue(np.isclose(np.linalg.norm(perp), 1.0))
            self.assertTrue(np.isclose(np.dot(perp, unit_vec), 0.0, atol=1e-10))
        # With uniform random phi, at least one sample should differ from the first.
        self.assertTrue(any(not np.allclose(results[0], sample) for sample in results[1:]))

    def test_get_perpendicular_fixed_phi_is_deterministic(self):
        vec = np.array([0.2, 0.6, -0.3])
        perp_1 = get_perpendicular(vec, phi=np.pi / 3.0)
        perp_2 = get_perpendicular(vec, phi=np.pi / 3.0)
        unit_vec = vec / np.linalg.norm(vec)
        self.assertTrue(np.allclose(perp_1, perp_2))
        self.assertTrue(np.isclose(np.linalg.norm(perp_1), 1.0))
        self.assertTrue(np.isclose(np.dot(perp_1, unit_vec), 0.0, atol=1e-10))

    def test_get_perpendicular_phi_zero_matches_expected_base_projection(self):
        vec_z = np.array([0.0, 0.0, 1.0])
        perp_z = get_perpendicular(vec_z, phi=0.0)
        self.assertTrue(np.allclose(perp_z, np.array([1.0, 0.0, 0.0])))

        vec_x = np.array([1.0, 0.0, 0.0])
        perp_x = get_perpendicular(vec_x, phi=0.0)
        self.assertTrue(np.allclose(perp_x, np.array([0.0, 1.0, 0.0])))

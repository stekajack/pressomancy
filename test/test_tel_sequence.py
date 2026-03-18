import numpy as np

from create_system import sim_inst, BaseTestCase
from pressomancy.object_classes.tel_sequence import TelSeq
from pressomancy.helper_functions import SinglePairDict, get_perpendicular, align_vectors


class _StubMonomer:
    simulation_type = SinglePairDict('stub_monomer', 999)

    def __init__(self):
        self.orientors = []

    def set_object(self, pos, ori):
        self.orientors.append(np.array(ori, dtype=float))
        return self


class TelSeqTest(BaseTestCase):
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        self.assertEqual(len(sim_inst.sys.part), 0)

    def test_antiparallel_uses_deterministic_phi_selector(self):
        monomers = [_StubMonomer(), _StubMonomer()]
        config = TelSeq.config.specify(
            espresso_handle=sim_inst.sys,
            n_parts=2,
            associated_objects=monomers,
            type='antiparallel',
        )
        instance = TelSeq(config=config)

        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        instance.set_object(pos=pos, ori=np.array([0.0, 0.0, 1.0]))

        chain_dir = np.array([0.0, 0.0, 1.0])
        phi_1 = instance._choose_antiparallel_phi(chain_dir)
        phi_2 = instance._choose_antiparallel_phi(chain_dir)
        self.assertTrue(np.isclose(phi_1, phi_2))
        expected_orientor = get_perpendicular(chain_dir, phi=phi_1)
        for monomer in monomers:
            self.assertEqual(len(monomer.orientors), 1)
            self.assertTrue(np.allclose(monomer.orientors[0], expected_orientor))

    def test_antiparallel_face_alignment_score_is_high(self):
        monomers = [_StubMonomer(), _StubMonomer()]
        config = TelSeq.config.specify(
            espresso_handle=sim_inst.sys,
            n_parts=2,
            associated_objects=monomers,
            type='antiparallel',
        )
        instance = TelSeq(config=config)
        z_axis = np.array([0.0, 0.0, 1.0])
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        test_dirs = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-2.0, 1.0, 0.5]),
            np.array([0.25, -1.0, 0.75]),
        ]
        for vec in test_dirs:
            chain_dir = vec / np.linalg.norm(vec)
            phi_opt = instance._choose_antiparallel_phi(chain_dir)
            side_axis = get_perpendicular(chain_dir, phi=phi_opt)
            rotation_matrix = align_vectors(z_axis, side_axis)
            x_world = rotation_matrix @ x_axis
            y_world = rotation_matrix @ y_axis
            score = max(np.abs(np.dot(x_world, chain_dir)), np.abs(np.dot(y_world, chain_dir)))
            self.assertGreater(score, 0.995)

    def test_parallel_and_hybrid_use_chain_orientation(self):
        for fold_type in ('parallel', 'hybrid'):
            monomers = [_StubMonomer(), _StubMonomer()]
            config = TelSeq.config.specify(
                espresso_handle=sim_inst.sys,
                n_parts=2,
                associated_objects=monomers,
                type=fold_type,
            )
            instance = TelSeq(config=config)
            pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
            instance.set_object(pos=pos, ori=np.array([0.0, 0.0, 1.0]))
            for monomer in monomers:
                self.assertEqual(len(monomer.orientors), 1)
                self.assertTrue(np.allclose(monomer.orientors[0], np.array([0.0, 0.0, 1.0])))
            sim_inst.reinitialize_instance()

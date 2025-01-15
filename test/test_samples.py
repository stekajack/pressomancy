import unittest
import importlib
import pkgutil
import warnings
import samples 
from create_system import sim_inst
from unittest import mock
from pressomancy.object_classes import Quadriplex
from pressomancy.helper_functions import MissingFeature
import gc

class SampleScriptTest(unittest.TestCase):
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        # gc.collect()

    def test_sample_scripts(self):
        def get_current_sim_instance(*args, **kwargs):
            """Helper function to return the latest `sim_inst`."""
            return sim_inst        
        for _, module_name, _ in pkgutil.iter_modules(samples.__path__):
            Quadriplex.fene_handle=None
            with self.subTest(script=module_name):
                with mock.patch("pressomancy.simulation.Simulation", side_effect=get_current_sim_instance):
                    try:
                        module = importlib.import_module(f"samples.{module_name}")
                    except MissingFeature:
                        warnings.warn(f"Skipping {module_name} because it requires a feature that is not available.")
                        continue
            sim_inst.reinitialize_instance()
            self.assertEqual(len(sim_inst.objects), 0)
            self.assertEqual(len(sim_inst.sys.part), 0)


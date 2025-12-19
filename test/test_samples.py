import importlib
import pkgutil
import logging
import samples 
import os
from create_system import sim_inst, BaseTestCase
from unittest import mock
from pressomancy.object_classes import Quadriplex
from pressomancy.helper_functions import MissingFeature


class SampleScriptTest(BaseTestCase):
    def tearDown(self) -> None:
        sim_inst.reinitialize_instance()
        sim_inst.sys.thermostat.turn_off()

    def test_sample_scripts(self):
        def get_current_sim_instance(*args, **kwargs):
            """Helper function to return the latest `sim_inst`."""
            return sim_inst        
        for _, module_name, _ in pkgutil.iter_modules(samples.__path__):
            Quadriplex.numInstances=0
            with self.subTest(script=module_name):
                with mock.patch("pressomancy.simulation.Simulation", side_effect=get_current_sim_instance):
                    try:
                        module = importlib.import_module(f"samples.{module_name}")
                    except MissingFeature as excp:
                        logging.warning(f"Skipping {module_name} because it requires a feature that is not available. Caught exception {excp}")
                        continue
                if os.getenv("PRESSOMANCY_TESTS_DUMP_VTF") in {"1", "true", "yes", "on"}:
                    from espressomd.io.writer import vtf
                    with open(f"test/{module_name}.vtf", mode="w+t") as fp:
                        vtf.writevsf(sim_inst.sys, fp)
                        vtf.writevcf(sim_inst.sys, fp)
                        fp.flush()
            sim_inst.reinitialize_instance()
            sim_inst.sys.thermostat.turn_off()
            sim_inst.sys.box_l = (100,100,100)
            self.assertEqual(len(sim_inst.objects), 0)
            self.assertEqual(len(sim_inst.sys.part), 0)
import importlib
import pkgutil
import logging
import samples 
import os
import sys
from create_system import sim_inst, BaseTestCase
from unittest import mock
from pressomancy.helper_functions import MissingFeature

class SampleScriptTest(BaseTestCase):
    
    def tearDown(self) -> None:
        self.cleanup()
        self.assertEqual(len(sim_inst.objects), 0)
        self.assertEqual(len(sim_inst.sys.part), 0)

    def test_sample_scripts(self):
        def get_current_sim_instance(*args, **kwargs):
            """Helper function to return the latest `sim_inst`."""
            if 'box_dim' in kwargs:
                sim_inst.sys.box_l = kwargs['box_dim']
            return sim_inst
        for _, module_name, _ in pkgutil.iter_modules(samples.__path__):
            full_module_name = f"samples.{module_name}"
            with self.subTest(script=module_name):
                with mock.patch("pressomancy.simulation.Simulation", side_effect=get_current_sim_instance):
                    try:
                        module = importlib.import_module(full_module_name)
                    except MissingFeature as excp:
                        logging.warning(f"Skipping {module_name} because it requires a feature that is not available. Caught exception {excp}")
                    else:
                        if os.getenv("PRESSOMANCY_TESTS_DUMP_VTF") in {"1", "true", "yes", "on"}:
                            from espressomd.io.writer import vtf
                            with open(f"test/{module_name}.vtf", mode="w+t") as fp:
                                vtf.writevsf(sim_inst.sys, fp)
                                vtf.writevcf(sim_inst.sys, fp)
                                fp.flush()
                    finally:
                        module = None
                        sys.modules.pop(full_module_name, None)
                        if hasattr(samples, module_name):
                            delattr(samples, module_name)
                        self.cleanup()
                        self.assertEqual(len(sim_inst.objects), 0)
                        self.assertEqual(len(sim_inst.sys.part), 0)

from pressomancy.simulation import Simulation
import logging
import unittest

class BaseTestCase(unittest.TestCase):

    box_dim=(50,50,50)
    min_global_cut=1
    
    @classmethod
    def setUpClass(cls):
        # Configure logging for tests
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

        if logger.hasHandlers():
            logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    def cleanup(self):
        # Reset the simulation instance after each test
        sim_inst.reinitialize_instance()
        sim_inst.sys.box_l=BaseTestCase.box_dim
        sim_inst.sys.min_global_cut=BaseTestCase.min_global_cut


sim_inst = Simulation(box_dim=BaseTestCase.box_dim)
sim_inst.set_sys(min_global_cut=BaseTestCase.min_global_cut)

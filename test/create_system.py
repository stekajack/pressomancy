from pressomancy.simulation import Simulation
import logging
import unittest

class BaseTestCase(unittest.TestCase):
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


sim_inst = Simulation(box_dim=(100,100,100))

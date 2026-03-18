from espressomd.io.writer import vtf
from espressomd import checkpointing
import time
from pressomancy.simulation import Simulation, GenericPart
import argparse
import sys as sysos
import os
import numpy as np
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# //////////////////////////////////////////////////////////////////////////////
# add custom input parameters for the simulation
parser = argparse.ArgumentParser()
parser.add_argument('-path_data', '--path_data', type=str, required=True,
                    help='absolute path to data')
parser.add_argument('-MODE', '--MODE', type=str, required=True,
                    help='start clean (NEW) or load from a checkpoint (LOAD_NEW)', choices=['NEW', 'LOAD', 'LOAD_NEW', 'INIT_SRC'])
parser.add_argument('-SRC_PATH', '--SRC_PATH', type=str, required=False,
                    help='path to the data file from which to initialise init config')
# add custom input parameters here such as particle size, time step, etc. For example:
# parser.add_argument('-n_part', '--n_part', type=int, required=True, default=10, help='number of particles in the system')
args = parser.parse_args()
parser_dict = vars(args).copy()
for key, value in parser_dict.items():
    if isinstance(value, list):
        parser_dict[key] = str(value).strip('[]').replace(', ', '')
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
# globals needed for basic simulation configuration
END_OF_TIME = time.time() + 259200  # 72h in seconds
STANDARD_ARG_DESTINATIONS = {'path_data', 'MODE', 'SRC_PATH'}
CONTEXT_STRING = '_'.join([
    str(val) for key, val in parser_dict.items() if key not in STANDARD_ARG_DESTINATIONS
])
TRAJ_PATH = os.path.join(args.path_data, f'trajectory_{CONTEXT_STRING}.vtf')
H5_DATA_PATH = os.path.join(args.path_data, f'custom_data_wip_{CONTEXT_STRING}.h5')
GLOBAL_COUNTER = None
EQUILIBRATION_ITERATIONS = 5
EQUILIBRATION_SEPARATION = int(2e05)
SAMPLING_ITERATIONS = 10
SNAPSHOT_SEPARATION = int(2000)
# //////////////////////////////////////////////////////////////////////////////

logging.info(CONTEXT_STRING)
logging.info(f'args.path_data: {args.path_data}')
logging.info(TRAJ_PATH)
logging.info(H5_DATA_PATH)

# //////////////////////////////////////////////////////////////////////////////
# Calculate parameters for the system such as box size, particle mass, etc. based on the input parameters
box_l = np.ones(3)  # box size in x, y, z directions
sim_inst = Simulation(box_dim=box_l)
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
# Set conditionals for system setup based on the input parameters such as time step, min_global_cut, skin, etc.
sim_inst.set_sys()
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
if args.MODE == 'INIT_SRC':
    assert args.SRC_PATH, 'MODE INIT_SRC requires that SRC_PATH is specified'
    CONTEXT_STRING_SRC = '_'.join([
        str(val) for key, val in parser_dict.items() if key not in STANDARD_ARG_DESTINATIONS
    ])
    mapping_dict = {'pos_ori_src_type': [], 'type_to_type_map': [], 'prop_to_prop_map': []}
    # mapping_dict['pos_ori_src_type'].append('real')
    # mapping_dict['type_to_type_map'].append(('real', 'yolk'))
    # mapping_dict['prop_to_prop_map'].append(('dip','dip'))
    sim_inst.set_init_src(
        path=os.path.join(args.SRC_PATH, f'custom_data_wip_{CONTEXT_STRING_SRC}.h5'), **mapping_dict)
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
if args.MODE == 'LOAD_NEW':
    logging.info(f'Loading checkpoint {CONTEXT_STRING} at {args.path_data}')
    checkpoint = checkpointing.Checkpoint(
        checkpoint_id=f"checkpoints_{CONTEXT_STRING}", checkpoint_path=args.path_data)
    try:
        checkpoint.load()
    except Exception:
        exc_type, value, traceback = sysos.exc_info()
        logging.info("Failed with exception [%s,%s ,%s]" %
            (exc_type, value, traceback))
        sim_inst.sys.part.clear()
        logging.info('Retrying to load')
        checkpoint.load()
    sim_inst.rebind_sys(sim_inst.sys)
    fp = open(TRAJ_PATH, mode='a')

else:

    # Setup initial configuration of the system
    objects_list=list()

    # //////////////////////////////////////////////////////////////////////////
    if args.MODE == 'INIT_SRC':
        sim_inst.set_prop_from_src(objects_list)
    # //////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////
    fp = open(TRAJ_PATH, mode='w+t')
    vtf.writevsf(sim_inst.sys, fp)
    vtf.writevcf(sim_inst.sys, fp)
    fp.flush()
    # //////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////
    checkpoint = checkpointing.Checkpoint(
        checkpoint_id=f"checkpoints_{CONTEXT_STRING}", checkpoint_path=args.path_data)
    checkpoint.register("sim_inst.sys")
    checkpoint.register("GLOBAL_COUNTER")
    logging.info("checkpoint registered")
    # //////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////
    for el in range(EQUILIBRATION_ITERATIONS):
        t0 = time.time()
        logging.info(f'EQUILIBRATION_ITERATIONS: {el}')
        sim_inst.sys.integrator.run(EQUILIBRATION_SEPARATION)
        t1 = time.time()
        logging.info(f"benchmark EQUILIBRATION_ITERATIONS: {t1-t0}")
        vtf.writevcf(sim_inst.sys, fp)
    # //////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
if GLOBAL_COUNTER is None:
    GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
        group_type=[GenericPart,], h5_data_path=H5_DATA_PATH, mode=args.MODE)
else:
    logging.info(f'GLOBAL_COUNTER was set to {GLOBAL_COUNTER} from previous load')
    GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
        group_type=[GenericPart,],
        h5_data_path=H5_DATA_PATH,
        mode=args.MODE,
        force_resize_to_size=GLOBAL_COUNTER)
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
if os.getenv("MK_SRC_MODE") in {"1", "true", "yes", "on"}:
    logging.info('MK_SRC_MODE environmental variable found and enabled the MK_SRC mode')
    assert args.SRC_PATH, 'MK_SRC_MODE requires that SRC_PATH is specified'
    assert args.SRC_PATH != args.path_data, 'path_data and src path must not be the same in MK_SRC mode. You are accidentally attempting to overwrite data!'
    sim_inst.mk_src_file(H5_DATA_PATH, os.path.join(args.SRC_PATH, f'custom_data_wip_{CONTEXT_STRING}.h5'), prop_dim=[('director', 3), ('image_box', 3)])
    logging.info('sucessfuly wrote SRC files and exited')
    sysos.exit(0)
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
benchmark_SAMPLING_INTERVAL = [0.,]
t1 = 0.
while GLOBAL_COUNTER < SAMPLING_ITERATIONS:
    t0 = time.time()
    if END_OF_TIME - t1 > benchmark_SAMPLING_INTERVAL[-1] * 2:
        logging.info(f"SAMPLING_ITERATIONS: {GLOBAL_COUNTER}")
        sim_inst.sys.integrator.run(SNAPSHOT_SEPARATION)
        vtf.writevcf(sim_inst.sys, fp)
        fp.flush()
        sim_inst.write_part_group_to_h5(time_step=GLOBAL_COUNTER)
        GLOBAL_COUNTER += 1
        if GLOBAL_COUNTER == SAMPLING_ITERATIONS:
            checkpoint.save()
            logging.info("final checkpoint.save() ran sucessfully. Exiting simulation")
    else:
        checkpoint.save()
        logging.info("checkpoint.save() ran sucessfully. Exiting loop.")
        break
    t1 = time.time()
    logging.info(f"time for sampling iteration {GLOBAL_COUNTER}: {t1 - t0}")
    benchmark_SAMPLING_INTERVAL.append(t1 - t0)
# //////////////////////////////////////////////////////////////////////////////
logging.info(f"benchmark SAMPLING_ITERATIONS: {np.mean(benchmark_SAMPLING_INTERVAL[1:])}")

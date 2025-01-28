from pressomancy.simulation import Simulation, RaspberrySphere, Filament
from pressomancy.helper_functions import BondWrapper
import espressomd
box_dim=(100,100,100)
sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys(timestep=0.001)
no_obj=20
part_per_fil=10
sigma=3
raspberries_config=RaspberrySphere.config.specify(sigma=sigma, size=sigma, espresso_handle=sim_inst.sys)
raspberries= [RaspberrySphere(config=raspberries_config) for x in range(no_obj)]
sim_inst.store_objects(raspberries)

grouped_raspberries = [raspberries[i:i+part_per_fil]
                for i in range(0, len(raspberries), part_per_fil)]
size=part_per_fil*3+(part_per_fil-1)*0.3*sigma

bond_hndl=BondWrapper(espressomd.interactions.FeneBond(k=10, d_r_max=3*sigma, r_0=0.83))
filament_config_list = [Filament.config.specify(sigma=sigma, n_parts=part_per_fil, size=size, espresso_handle=sim_inst.sys, bond_handle=bond_hndl, associated_objects=rsp) for rsp in grouped_raspberries]
filaments = [Filament(config=rsp) for rsp in filament_config_list]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)


for filament in filaments:
    filament.bond_nearest_part(type_key='virt')

for rasp in raspberries:
    rasp.set_hydrod_props(rot_inertia=43,mass=47.77)

lbb = sim_inst.init_lb(kT=0.1, agrid=1, dens=1, visc=1, gamma=6.23)
sim_inst.create_flow_channel()
sim_inst.sys.integrator.run(0)

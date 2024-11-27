from pressomancy.simulation import Simulation, RaspberrySphere, Filament
import espressomd
import os
import sys as sysos
import numpy as np
import time
from espressomd.io.writer import vtf
box_dim=(100,100,100)
sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys(timestep=0.001)
no_obj=20
part_per_fil=10
simga=3
raspberries= [RaspberrySphere(sigma=simga, espresso_handle=sim_inst.sys) for x in range(no_obj)]
for rasp in raspberries:
    rasp.set_hydrod_props(rot_inertia=43,mass=47.77)

sim_inst.store_objects(raspberries)
grouped_raspberries = [raspberries[i:i+part_per_fil]
                for i in range(0, len(raspberries), part_per_fil)]
size=part_per_fil*3+(part_per_fil-1)*0.3*simga

bond_hndl=espressomd.interactions.FeneBond(k=10, d_r_max=3*simga, r_0=0.83)
sim_inst.sys.bonded_inter.add(bond_hndl)

filaments = [Filament(sigma=simga, n_parts=part_per_fil, size=size, espresso_handle=sim_inst.sys,associated_objects=rsp) for rsp in grouped_raspberries]
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)


for filament in filaments:
    filament.bond_nearest_part(bond_handle=bond_hndl,type_key='virt')

lbb = sim_inst.init_lb_GPU(kT=0.1, agrid=1, dens=1, visc=1, gamma=6.23)
sim_inst.create_flow_channel()
sim_inst.sys.integrator.run(0)
fp = open('trajectory.vtf', mode='w+t')
vtf.writevsf(sim_inst.sys, fp)
vtf.writevcf(sim_inst.sys, fp)

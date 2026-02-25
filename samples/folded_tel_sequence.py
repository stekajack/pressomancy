from pressomancy.simulation import Simulation, TelSeq, Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper
import espressomd
import numpy as np
import logging
# length scale 4e-10
# mass scale 2.5727522264874994e-20
# energy scale 4.116403562379999e-21
# time scale 1e-09
N_avog = 6.02214076e23
sigma = 1.
rho_si = 0.6*N_avog
no_obj=18
N = int(no_obj/3)
vol = N/rho_si
box_l = pow(vol, 1/3)
_box_l = box_l/0.4e-09
box_dim = _box_l*np.ones(3)

Rf=2e-10
etaw = 0.87e-3
t_=1e-9
d_=4e-10
mass_ = 2.5727522264874994e-20
gamma_T = 6*np.pi*etaw*Rf*(t_/mass_)
gamma_R = 8*np.pi*etaw*pow(Rf, 3)*(t_/(pow(d_, 2)*mass_))
print("gamma_T: ", gamma_T)
print("gamma_R: ", gamma_R)

sheets_per_quad = 3
part_per_filament = 2

sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys(timestep=0.005)
logging.info(f'box_dim: {sim_inst.sys.box_l}')
bond_hndl=BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=1.5))
quartet_config = Quartet.config.specify(bond_handle=bond_hndl,type='broken', espresso_handle=sim_inst.sys)
quartets = [Quartet(config=quartet_config) for x in range(no_obj)]

grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=sim_inst.sys, bonding_mode='ftf',bond_handle=bond_quad, size=np.sqrt(3)*5) for elem in grouped_quartets]
quadriplex = [Quadriplex(config=elem) for elem in quadriplex_config_list]

diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=np.sqrt(2)*4.2, d_r_max=2*1.5))
across_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=4.2, d_r_max=2*1.5))
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
fold_types = ['parallel', 'hybrid', 'antiparallel']
tel_config_list = [
    TelSeq.config.specify(
        n_parts=part_per_filament,
        espresso_handle=sim_inst.sys,
        associated_objects=grouped_quadriplexes[idx],
        size=quadriplex[0].params['size'] * part_per_filament + np.sqrt(3) * bond_quad.r_0 + (part_per_filament - 1),
        bond_handle=bond_quad,
        diag_bond_handle=diag_bond,
        across_bond_handle=across_bond,
        spacing=6.,
        type=fold_type,
    )
    for idx, fold_type in enumerate(fold_types)
]
telomeres = [TelSeq(config=elem) for elem in tel_config_list]
sim_inst.store_objects(telomeres)
sim_inst.set_objects(telomeres)

for telomere in telomeres:
    telomere.wrap_into_Tel()
sim_inst.sys.integrator.run(0)
for quartet in quartets:
    quartet.add_h_bond_patches()

angle_harmonic = espressomd.interactions.AngleHarmonic(bend=10.0, phi0=np.pi)
sim_inst.sys.bonded_inter.add(angle_harmonic)
for quadriplex in quadriplex:
    quadriplex.add_bending_potential(angle_harmonic)
    quadriplex.add_dihedrals()
    quadriplex.add_extra_bendings()

sim_inst.set_steric_custom(
    pairs=[ ('real', 'real'),
            ('real', 'virt'),
            ('real', 'circ'), 
            ('real', 'cation'), 
            ('virt', 'circ'),
            ('virt', 'cation'),
            ('circ', 'circ'),
            ('cation', 'cation'),
            ('virt', 'virt')],
            wca_eps=[1, 1, 1, 1, 1, 1, 1, 1, 1], sigma=[0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87])
# sim_inst.set_vdW_custom(
#     pairs=[('squareA', 'squareB'),], lj_eps=[5,], lj_sigma=[0.4,], r_min=0.4)

sim_inst.sys.non_bonded_inter[sim_inst.part_types['squareA'], sim_inst.part_types['squareB']].morse.set_params(eps=5.0,alpha=4.8,rmin = 0.4, cutoff=1.5)

sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=gamma_T,
                               gamma_rotation=gamma_R, seed=sim_inst.seed)

electrostatidcs_solver=espressomd.electrostatics.DH(prefactor = 1., kappa=0, r_cut=8)
# ,check_neutrality=False
sim_inst.sys.electrostatics.solver = electrostatidcs_solver
sim_inst.sys.integrator.run(0)
energy = sim_inst.sys.analysis.energy()

for keys,val in energy.items():
    if val!=0:
        print(keys,val)

from espressomd.io.writer import vtf
with open(f"/home/stekajack/DATA_VIEW/quadriplex/sim_test/folded_tel_compulsion.vtf", mode="w+t") as fp:
    vtf.writevsf(sim_inst.sys, fp)
    vtf.writevcf(sim_inst.sys, fp)
    for _ in range(2000):
        sim_inst.sys.integrator.run(1)
        vtf.writevcf(sim_inst.sys, fp)
        fp.flush()
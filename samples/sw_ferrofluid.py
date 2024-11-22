import numpy as np
from pressomancy.simulation import Simulation, SWPart


n_part=100
vol_frac=0.01
xi=5
kT_KVm_inv=5
tstp=0.001
LJ_SIGMA = 1
LJ_EPSILON = 1
LJ_CUT = 2**(1. / 6.) * LJ_SIGMA

temperature = 1

# GENERAL UNITS FOR THE SIMULATIONS

# kg m /(s^2 A^2)
mu0 = 4.e-7*np.pi
# m^2 kg /(s^2 K)
kb = 1.38064852e-23
T = 298.15
# dynamic viscosity of background fluid (let's assume water at room T) [kg / (m s)]
# etaw = 0.89e-3
# saturation magnetization of magnetite, A/m (480 is used frequently by theoreticians). Experimental measurements: https://dx.doi.org/10.1103/PhysRevE.75.051408
Mm = 480.e3
# magnetic aisotropy constant          [kg / (m s^2)]
K = 1.e4
Ha = 2 * K / (Mm*mu0)  # anisotropy field                     [A / m]
gyrm_rat = mu0 * 1.76 * 10**11
alpha = 0.08  # Gilbert damping parameter
omega_l = gyrm_rat * Ha
# density of magnetite, kg/m^3
densm = 5170.0
# density of oleic acid, coating magnetite, kg/m^3.
densmsh = 895.0

Dm = pow(6*kT_KVm_inv*kb*T/(np.pi*K), 1/3)
# thickness of oleic acid coating, m
Rmsh = 2.e-9
Rm = Dm/2.                                                                # radius of the magnetic core, m
# total radius, m
Rf = Rm+Rmsh
Vm = np.pi * (Dm**3.) / 6.
dip = Mm*Vm
Vtot = (4./3.)*np.pi*Rf**3
massm = (4./3.)*np.pi*(densm*(Rm**3.) + densmsh *
                    ((Rf)**3. - Rm**3.))         # mass, kg

# UNIT SCALE

colloid_radius_MD = 0.5
# length scale, m
d_ = Rf/colloid_radius_MD
# mass scale, kg
mass_ = massm
# energy scale, m^2 kg / s^2 (J)
U_ = kb*T
# time scale, s
t_ = d_*np.sqrt(mass_/U_)

tau_d = kT_KVm_inv/omega_l * (1 + alpha**2)/alpha

etaw = tau_d/(3*0.02*Vtot / (kb*T))
tau_b = 3 * etaw * Vtot / (kb*T)
tau_n = tau_d * (np.exp(kT_KVm_inv) - 1)/(2 * kT_KVm_inv)/(1/(1 + 1/kT_KVm_inv)
                                                                  * (kT_KVm_inv/np.pi)**0.5 + 2**(-kT_KVm_inv - 1))
print("t_d/t_b: ", tau_d/tau_b)
dip_ = np.sqrt(4.*np.pi*U_*(d_**3.)/(mu0))
A_ = dip_/(d_*d_)

# H-field scale, A/ m (SW needs this to be passed!)
H_ = A_/d_

# REDUCED PARAMETERS TO PASS TO ESPRESSO
x_dot_H_ = xi*kb*T/(mu0*dip)
H_reduced = mu0 * x_dot_H_ * A_ * t_ * t_ / mass_
HK_inv = 1/(Ha*mu0 * A_ * t_ * t_ / mass_)
dip_reduced = dip/dip_
gamma_T = 6*np.pi*etaw*Rf*(t_/mass_)
gamma_R = 8*np.pi*etaw*pow(Rf, 3)*(t_/(pow(d_, 2)*mass_))
SAMPLING_ITERATIONS = 100
SNAPSHOT_SEPARATION = int(max(tau_b, tau_n)/(tstp*t_))
print('Core size in SI for reference', Dm)
print('H', H_reduced)
print("kT_KVm_inv: ", kT_KVm_inv)
print("HK_inv: ", HK_inv)
print('h: ', H_reduced*HK_inv)
print("dipole moment, (mu)*: ", dip_reduced)
print('time scale', t_)
print('browninan relaxation time ', tau_b)
print('debeye relaxation time ', tau_d)
print('neel relaxation time ', tau_n)
print('SNAPSHOT_SEPARATION ', SNAPSHOT_SEPARATION)
print("translational friction , (gamme_T): ", gamma_T)
print("rotational friction , (gamme_R): ", gamma_R)

tau0_inv = 1/(mu0*Mm/(2*gyrm_rat*K) *
                                (1+alpha**2)/alpha*np.sqrt(np.pi*kb*T/(K*Vm)))
box_l = np.power((4*n_part*np.pi*np.power(LJ_SIGMA/2, 3)) /
                 (3*vol_frac), 1/3)*np.ones(3)
print("box_l ", box_l)

sim_inst = Simulation(box_dim=box_l)
sim_inst.set_sys()
# add real particles that carry can rotate, and carry the kT_KVm_inv [SI] and dt_incr (time elapsed in one integration in [SI] )
sw_parts=[SWPart(sigma=LJ_SIGMA,kT_KVm_inv=kT_KVm_inv,dipm=dip_reduced,dt_incr=t_*tstp,tau0_inv=tau0_inv,HK_inv=HK_inv,espresso_handle=sim_inst.sys) for _ in range(n_part)]
sim_inst.store_objects(sw_parts)
sim_inst.set_objects(sw_parts)

sim_inst.set_steric(key=('sw_real', ), wca_eps=LJ_EPSILON)
sim_inst.set_H_ext(H=(0, 0, H_reduced))

sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=gamma_T,
                               gamma_rotation=gamma_R, seed=sim_inst.seed)

sim_inst.sys.integrator.run(0)

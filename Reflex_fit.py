from kinematics_halo import data
import numpy as np
from pymultinest.solve import solve
import json
import sys

"""
Important:

spherical unit convension:
Phi   = azimuthal angle
theta = polar angle
"""
vsun_mw = np.array([11.1, 244.24, 7.25]) #km/s
rsun_mw = np.array([-8.3, 0., 0.02]) #kpc


#velcity in spherical coords is given by
def cartesian_to_spherical(x,y,z,vx,vy,vz):
    phi = np.arctan2(y,x)
    th = np.arccos(z/np.sqrt(x**2 + y**2+ z**2))
    r = np.sqrt(x**2. + y**2. + z**2.)

    #Code part from Mpetersen
    cost = z/r
    sint = np.sqrt(1. - cost *cost)
    cosp = np.cos(phi)
    sinp = np.sin(phi)

    vr = sint * (cosp * vx + sinp * vy) + cost * vz
    vphi = (-sinp * vx + cosp * vy)
    vtheta = (cost * (cosp * vx + sinp * vy) - sint * vz)

    return np.array([r,phi, th]), np.array([vr, vphi, vtheta])

def euler_xyz(phi,theta,psi=0., deg=False):
    #FUNCTION FROM MIKE PETERSEN GITHUB
    #https://github.com/michael-petersen/ReflexMotion/blob/master/reflexmotion/reflex.py
    if deg == True:
        phi, theta = np.deg2rad(phi), np.rad2deg(theta)

    Rmatrix = np.array([[ np.cos(theta)*np.cos(phi),\
                          np.cos(theta)*np.sin(phi),\
                         -np.sin(theta)],\
                        [np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi),\
                         np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi),\
                         np.cos(theta)*np.sin(psi)],\
                        [np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi),\
                         np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi),\
                         np.cos(theta)*np.cos(psi)]])
    return Rmatrix

def add_vtravel(vtravel, phi2):
    """

    :param vtravel: travel velocity
    :param phi2: inclination angle in rotated frame
    :return:
    v : vector of spherical velocity components in rotated frame.
    """

    vr = -vtravel*np.cos(phi2)
    vphi2 = +vtravel*np.sin(phi2)
    vphi1 = 0.
    return np.array([vr, vphi1, vphi2])

def spherical_unit_vectors(phi,th):
    """

    :param phi: azimuthal angle, can be array of 1xN
    :param th: inclination angle, can be array of 1xN
    :return: er, ephi, eth : unit vectors of shape Nx3
    """
    er = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), (np.cos(th))])
    ephi = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    eth = np.array([np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)])
    return er, ephi, eth


def spherical_to_cartesian(r,phi,th, vr, vphi, vth):
    ur   = np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), (np.cos(th))])
    uphi = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(r)])
    uth  = np.array([np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), -np.sin(th)])

    rcart = np.array(r*ur)

    vx = vr * ur[0] + vth * uth[0] + vphi * uphi[0]
    vy = vr * ur[1] + vth * uth[1] + vphi * uphi[1]
    vz = vr * ur[2] + vth * uth[2] + vphi * uphi[2]

    vcart = np.array([vx,vy,vz])

    return rcart, vcart

#1. Define the galactocentric coordinates
d = np.loadtxt("mocks/mock_live_s13_98_35k_galactocentric_outer_footprint.txt").T
x, y, z    = d[:,0], d[:,1], d[:,2]
vx, vy, vz = d[:,3], d[:,4], d[:,5]

rgal = np.zeros((len(x),3))
rgal[:,0] = x
rgal[:,1] = y
rgal[:,2] = z

vgal = np.zeros((len(x), 3))
vgal[:,0] = vx
vgal[:,1] = vy
vgal[:,2] = vz

#2. compute spherical galactic coordinates
rgalsph = np.zeros((len(x),3))
vgalsph = np.zeros((len(x),3))
for i in range(len(x)):
    tmp = cartesian_to_spherical(rgal[i,0], rgal[i,1], rgal[i,2],
                                  vgal[i,0], vgal[i,1], vgal[i,2])
    rgalsph[i,:] = tmp[0]
    vgalsph[i,:] = tmp[1]


#3. rotate cartesian coordinates such that the z axis points at lapex, bapex
# through the euler angle rotation x-y-z

def get_v(cube, rgal, vgal):

    Rrot = euler_xyz(cube[0], np.pi/2. - cube[1], deg=False)
    rp   = np.zeros_like(rgal)
    vp   = np.zeros_like(vgal)

    for i in range(len(x)):
        rp[i,:] = np.dot(Rrot,rgal[i,:])
        vp[i, :] = np.dot(Rrot, vgal[i, :])

    #4. compute the spherical coordinates of r,v in rotated frame

    rpsph = np.zeros_like(rp)
    vpsph = np.zeros_like(vp)

    for i in range(len(x)):
        tmp = cartesian_to_spherical(rp[i, 0], rp[i, 1], rp[i, 2],
                                     vp[i, 0], vp[i, 1], vp[i, 2])
        rpsph[i, :] = tmp[0]
        vpsph[i, :] = add_vtravel(cube[2], rpsph[i,2])# + tmp[1]

    #5. translate the spherical coordinates in the rotated frame back to cartesian coordinates
    # note: we don't care about the position vector anymore, we only want the velocities

    rp1 = np.zeros_like(rp)
    vp1 = np.zeros_like(rp)
    for i in range(len(x)):
        temp = spherical_to_cartesian(rpsph[i, 0], rpsph[i, 1], rpsph[i, 2],
                                     vpsph[i, 0], vpsph[i, 1], vpsph[i, 2])
        rp1[i,:] = temp[0]
        vp1[i,:] = temp[1]

    #6. now rotate the coordinates back to the galactic frame

    rgal1 = np.zeros_like(rgal)
    vgal1 = np.zeros_like(rgal)
    for i in range(len(x)):
        rgal1[i,:] = np.dot(Rrot.T,rp1[i,:])
        vgal1[i, :] = np.dot(Rrot.T, vp1[i, :])


    #7. compute the vector v and translate to be at the sun
    r = np.zeros_like(rp)
    v = np.zeros_like(vp)

    for i in range(len(x)):
        r[i,:] = rgal1[i,:] - rsun_mw
        v[i,:] = vgal1[i,:]  - vsun_mw + np.array([cube[3], cube[4], cube[5]])

    #8. find the galactic proper motions using the unit vectors of ul, ub, ulos
    # spherical unit vectors in the galactic frame..
    rsunsph = np.zeros_like(r)
    vsunsph = np.zeros_like(v)

    for i in range(len(x)):
        temp = cartesian_to_spherical(r[i,0], r[i,1], r[i,2],
                                         v[i,0], r[i,1], r[i,2])
        rsunsph[i,:] = temp[0]
        vsunsph[i,:] = temp[1]

    #now multiply the vector v by the unit vectors in spherical coordinates centred at
    #the positions of the sun

    vlos = np.zeros_like(r)
    mul = np.zeros_like(r)
    mub = np.zeros_like(r)

    elos, ephi, eth = spherical_unit_vectors(rsunsph[:,1], rsunsph[:,2])
    elos, ephi, eth = elos.T, ephi.T, eth.T

    #helicocentric distance
    dist = np.linalg.norm(rsunsph,axis=1)
    fac = 4.74057*dist
    vtest = vgal1 - vsun_mw
    for i in range(len(x)):

        vlos[i,:]      = np.dot(v[i,:], elos[i,:])
        mul[i,:]       = np.dot(vtest[i,:], ephi[i,:])/fac[i]
        mub[i,:]       = -np.dot(vtest[i,:], eth[i,:])/fac[i]

    return vlos[:,0], mul[:,0], mub[:,0]

#9. Define the likelihood functions for the perturbed velocities

def like_vlos(cube, vlos_data, vlos_param, evlos):
    """

    :param vlos_data: measured los vel. s.v.
    :param vlos_param: los. vel. with vtravel s.v.
    :param evlos: obs. errors s.v.
    :param sigvlos: vlos hyperparameter, cube
    :return:
    """
    sigvlos = 1./cube[6]
    evlos2 = evlos**2. + sigvlos**2.
    exponent = (1./evlos2)*((vlos_data - vlos_param)**2.)
    ln = np.log(2*np.pi*evlos2)
    plos = -0.5*(ln + exponent)

    return np.exp(plos)

def like_pms(cube,pml_data,pmb_data, dist, corr, epml, epmb, edist, pml_param, pmb_param):
    """

    :param cube: Multinest hypercube (1xNparam)
    :param pml_data: observational proper motion in l s.v.
    :param pmb_data: observational proper motion in b s.v.
    :param dist: observed heliocentric distance s.v.
    :param corr: proper motion correlation s.v.
    :param epml: proper motion uncertainty s.v.
    :param epmb: proper motion uncertainty s.v.
    :param edist: observational distance error s.v.
    :param pml_param: proper motion with vtravel in l s.v.
    :param pmb_param: proper motion with vtravel in b s.v.
    :return:
    """

    sigpml = 1./cube[7]
    sigpmb = 1./cube[8]
    fac = 4.74057*dist
    elp2 = epml ** 2. + edist ** 2. * (np.abs(pml_data) ** 2. / dist ** 2) + (sigpml ** 2. / fac ** 2.)
    ebp2 = epmb ** 2. + edist ** 2. * (np.abs(pmb_data) ** 2. / dist ** 2) + (sigpmb ** 2. / fac ** 2.)

    cov = np.array([[elp2, epml*epmb*corr],
                    [epml*epmb*corr, ebp2]])

    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)

    kl = pml_data - pml_param
    kb = pmb_data - pmb_param

    v1 = kl*inv[0,0] + kb*inv[0,1]
    v2 = kl*inv[1,0] + kb*inv[1,1]

    exp = kl*v1 + kb*v2
    ln  = np.log(((2*np.pi)**2.)*det)

    ppm = -0.5*(ln + exp)

    return np.exp(ppm)

#10. compute the likelihood function for the data and model..

def LogLikelihood(cube):
    #np.array([l,b,vtravel, vr,vphi,vth,svlos,sl,sb])

    lnptot  = 0.

    #x [kpc]; y [kpc]; z [kpc]; vx [km/s]; vy [km/s]; vz [km/s];
    # dL [deg]; dB [deg]; dist [kpc]; vlos [km/s]; dmul [marcsec/yr]; dmub [marcsec/yr];
    # edist[kpc]; evlos [km/s]; edmul [marcsec/yr]; edmub [marcsec/yr];
    # rapo [kpc]; rapo quality; sdss?

    #pml_data,pmb_data, dist, corr, epml, epmb, edist, pml_param, pmb_param, sigpml, sigpmb

    vlos, mul, mub = get_v(cube, rgal, vgal)
    for i in range(len(x)):
        a = like_vlos(cube, d[i,9], vlos[i], d[i,13]) + \
            like_pms(cube, d[i,10],d[i,11],d[i,8], 0.0, d[i,14], d[i,15], d[i,12], mul[i], mub[i])

        loga = np.log(a)
        lnptot += loga

    return lnptot


def Prior(cube):
    # a + (b-a) * cube rane a < x < b
    #Order:  l,b,vtravel, vr,vphi,vth,svlos,sl,sb
    cube[0] =  - np.pi + cube[0]*2*np.pi   # |L|
    cube[1] =  - np.pi/2. + cube[1]*2*np.pi/2.  # phi
    cube[2] = 1. + cube[2] * 100.  # theta
    cube[3] = 1. + cube[3] * 100.  # theta
    cube[4] = 1. + cube[4] * 100.  # theta
    cube[5] = 1. + cube[5] * 100.  # theta
    cube[6] = (1./1000.)**2 + cube[6]*((1./10.)**2 - ((1./1000.)**2))  # errx
    cube[7] = (1./1000.)**2 + cube[7]*((1./10.)**2 - ((1./1000.)**2))   # erry
    cube[8] = (1./1000.)**2 + cube[8]*((1./10.)**2 - ((1./1000.)**2))   # errz

    return cube

#Running the sampler
parameters = ["l", "b", "vtravel", "vr", "vphi", "vth", "sigvlos",
            "sigmul", "sigmub"]
prefix = "chains"

n_params = len(parameters)
result = solve(LogLikelihood=LogLikelihood, Prior=Prior,
               n_dims=n_params, outputfiles_basename=prefix, verbose=True,
               resume=False, n_live_points=100, wrapped_params=None)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
    print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)
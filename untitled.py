from kinematics_halo import data
import numpy as np
import matplotlib.pyplot as plt

"""
Important:

spherical unit convension:
Phi   = azimuthal angle
theta = polar angle
"""

def make_data():
    cuts = [True, False]
    names = ["outer", "inner"]
    sdsscuts = ["footprint", "allsky"]

    #make 2 dfs, 1 with allsky and no r-cut and another with r-cut at 20kpc and SDSS foot
    #print
    c=0
    for cut in cuts:
        a = data("mocks/mock_live_s13_98_35k_heliocentric.txt",
                 dataset="halo", rcut=cut, SDSS=cut)
        a.posvel()
        a.errors()
        a.save_df("mocks/mock_live_s13_98_35k_galactocentric_%s_%s.txt"%(names[c], sdsscuts[c]))
        c+=1
    return

def get_fake_params():
    """
    function to generate fake model parameters.
    :return:
    lapex, bapex,
    """
    randarray3x3= np.random.randn(3)*10.
    #Apex longitude/lattitude [deg]
    l,b, vtravel = 50, -30, 20
    #Mean halo spherical velocities [km/s]
    vr,vphi,vth = randarray3x3[0], randarray3x3[1], randarray3x3[2]
    #Vlos hyperparameter
    svlos = np.random.randn()*30.
    #Galactic longitude/lattide velocity hyperparameters
    sl, sb = randarray3x3[0]*0.2, randarray3x3[1]*0.7

    return np.array([l,b,vtravel, vr,vphi,vth,svlos,sl,sb])

#http://www.astrosurf.com/jephem/library/li110spherCart_en.htm
def rdot(x,y,z,vx,vy,vz):
    return (x*vx + y*vy + z*vz)/(np.sqrt(x**2. + y**2. + z**2.))
def thdot(x,y,z,vx,vy,vz):
    #azimuth
    return (vx*y - vy*x)/(x**2. + y**2.)
def phidot(x,y,z,vx,vy,vz):
    #inclination
    return (z*(x*vx + y*vy) - (x**2. + y**2.)*vz)/((x**2. + y**2. +z**2.)*np.sqrt(x**2 + y**2.))

#velcity in spherical coords is given by
def cartesian_to_spherical(x,y,z,vx,vy,vz):
    rdott = rdot(x,y,z,vx,vy,vz)
    thdott = thdot(x,y,z,vx,vy,vz)
    phidott = phidot(x,y,z,vx,vy,vz)

    phi = np.arctan2(y,x)
    th = np.arccos(z/np.sqrt(x**2 + y**2+ z**2))
    r = np.sqrt(x**2. + y**2. + z**2.)
    return np.array([r,phi, th]), np.array([rdott, r*thdott*np.sin(th), r*phidott])

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
    uphi = np.array([-np.sin(phi), np.cos(phi), 0.])
    uth  = np.array([np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), -np.sin(th)])

    rcart = np.array(r*ur)

    vx = vr * ur[0] + vth * uth[0] + vphi * uphi[0]
    vy = vr * ur[1] + vth * uth[1] + vphi * uphi[1]
    vz = vr * ur[2] + vth * uth[2] + vphi * uphi[2]

    print(ur)
    vcart = np.array([vx,vy,vz])

    return rcart, vcart

#1. Define the galactocentric coordinates
d = np.loadtxt("mocks/mock_live_s13_98_35k_galactocentric_outer_footprint.txt").T
x, y, z    = d[:,0], d[:,1], d[:,2]
vx, vy, vz = d[:,3], d[:,4], d[:,5]

vsun_mw = np.array([11.1, 244.24, 7.25]) #km/s
rsun_mw = np.array([-8.3, 0., 0.02]) #kpc

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

params = get_fake_params()


Rrot = euler_xyz(params[0], np.pi/2. -params[1], deg=True)
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
    vpsph[i, :] = tmp[1] #+ add_vtravel(params[2], rpsph[i,2])

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
    v[i,:] = vgal1[i,:] + np.array([params[3], params[4], params[5]]) - vsun_mw

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
    mub[i,:]       = np.dot(vtest[i,:], eth[i,:])/fac[i]










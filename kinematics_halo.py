import pandas as pd
import numpy as np
from sys import exit
from scipy.stats import spearmanr
from alive_progress import alive_bar
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
import astropy.coordinates as coord
import astropy.units as u

class Sagittarius(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Sagittarius dwarf galaxy, as described in
        https://ui.adsabs.harvard.edu/abs/2003ApJ...599.1082M
    and further explained in
        https://www.stsci.edu/~dlaw/Sgr/.

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_Lambda_cosBeta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : `~astropy.units.Quantity`, optional, keyword-only
        The radial velocity of this object.
        "Using the Majewski euler angles"
        "code from https://www.stsci.edu/~dlaw/Sgr/"

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]
    }


class data:
    def __init__(self, fname, *args, **kwargs):
        self.fname = fname

        self.dataset = None
        #Numpy data arrays
        self.dat   = None

        #Pandas dfs
        self.dfsim = None
        self.df    = None

        self.N  = None
        #Galactic coordinates, distances and error
        self.d     = None
        self.l     = None
        self.b     = None
        self.pma   = None
        self.pmd   = None
        self.epma  = None
        self.epmd  = None
        self.ed    = None
        self.vlos  = None
        self.evlos = None

        #Cartesian coodinates and angular momenta
        #Gravity Collaboration 2018, Bennet, Bovy 2019, 30pc in z
        self.r_sun_mw  = np.array([-8.3, 0.0, 0.03])
        self.v_sun_mw  = np.array([11.1, 244.24, 7.25]) #Changed to be consistent with Jorge
        self.sgr_l     = np.array([431., -4919., -778.]) #Pennarubia et al. 2010
        self.T = np.array([[- 0.06699, - 0.87276 ,- 0.48354],   #for ra-dec-pmra-pmdec input
                           [+ 0.49273, - 0.45035, + 0.74458],
                           [- 0.86760, - 0.18837, + 0.46020]])
        self.x  = None
        self.y  = None
        self.z  = None
        self.ex = None
        self.ey = None
        self.ez = None

        self.vx = None
        self.vy = None
        self.vz = None
        self.evx = None
        self.evy = None
        self.evz = None

        self.lx  = None
        self.ly  = None
        self.lz  = None
        self.elx = None
        self.ely = None
        self.elz = None

        self.sgrb= None
        #correlation coefficients of error points
        self.cxy = None
        self.cxz = None
        self.cyz = None
        self.lc  = None

        #Spherical coordinates and angular momenta
        self.r    = None
        self.th   = None
        self.phi  = None
        self.vr   = None
        self.vth  = None
        self.vphi = None

        self.lr   = None
        self.lth  = None
        self.lphi = None
        self.ls   = None

        self.rhat   = None
        self.thhat  = None
        self.phihat = None

        #additonal vars
        self.SDSS    = None
        self.rcut    = None
        self.split   = None
        self.ndraw   = None  #Draw subset of stars from sample.
        self.f       = None
        self.bcut  = None
        self.radec = False
        #positions of stars in the frame of sgr
        self.prog_r = None
        self.prog_l = None
        self.progb  = None
        self.bcut   = None
        self.bcut_method = "angmom"


        for key, val in kwargs.items():
            print(key,val)
            if key == "SDSS":
                self.SDSS = val
            elif key == "rcut":
                self.rcut = val
            elif key == "dataset":
                self.dataset = val
            elif key == "split":
                self.split = val
            elif key == "draw":
                self.ndraw = val
            elif key == "bcut":
                self.bcut = val
            elif key == "b_method":
                self.bcut_method = val
            elif key == 'radec':
                self.radec = val
            elif key == 'angmom_prog':
                self.prog_l = val


        self.readdata()

        if self.split == True:
            self.split_sample()

        self.makedfs(True)
        self.sample_draw()
        self.setvars()

    def posvel(self, frame=True):
        self.get_cpos_icrs()
        self.get_v_icrs()


        if self.bcut_method == 'Majewski':
            self.sgr_b_majewski()
        elif self.bcut_method == 'angmom':
            self.prog_b(self.prog_l)

        if frame == True:
            self.convert_icrs_mw()
            self.convert_icrs_mw_vel()
            self.cangmom()
            return print("Position and Velocity in GCC Frame")
        return print("Position and velocity computed")

    def errors(self):
        self.cposerr_icrs()
        self.v_icrs_errs()
        self.cangmom_err()
        return


    def readdata(self):
        self.dat = np.loadtxt(self.fname, skiprows=1)

        if self.dataset == 'halo':
            self.header_sim = ["x", "y", "z", "vx", "vy", "vz",
                           "l", "b", "dist", "vlos", "dmu_l", "dmu_b",
                           "edist", "evlost", "edmu_l", "edmu_b",
                           "rapo", "quality", "sdss"]


        elif self.dataset == 'GCcat':
            self.header_sim = ["x", "y", "z", "vx", "vy", "vz", "l", "b", "dist", "vlos", "dmu_l", "dmu_b", "edist",
                               "evlost", "edmu_l", "edmu_b", "dead", "dead","dead",
                               "Sgr_lambda", "Sgr_beta", "Belokurov Flag", "corr"]

        return print("Data reading complete")


    def split_sample(self):
        """
        This method splits the mock dataset into half,
        """
        A, B = np.split(self.dat, 2)
        self.dat = A

        return print("Sample split into 2, size = %3.2f"%(len(A)))

    def sample_draw(self):
        #Compute fraction of gse/sgr stars in sample to maintain.
        if self.dataset=='halo':
            self.f =1

        if (self.ndraw is not None) & (self.dataset == 'halo'):
            tmpdf = self.df[self.df[self.dataset] == 0] #choose halo stars only
            print(len(tmpdf))
            self.df = tmpdf.sample(n=self.ndraw)

        else:
            print("No samples drawn")
        return

    def makedfs(self, cut):
        self.dfsim = pd.DataFrame(self.dat, columns=self.header_sim)

        if self.rcut == True and self.SDSS == True:
            ##cut r > 20 kpc
            self.df = self.dfsim[np.sqrt(self.dfsim['x']**2. + \
            self.dfsim['y']**2. + self.dfsim['z']**2.) > 20.]
            #cut SDSS footprint
            self.df = self.df[self.df['sdss'] == 1.]
        elif self.rcut == True and self.SDSS == False:
            ##cut r > 20 kpc
            self.df = self.dfsim[np.sqrt(self.dfsim['x']**2. + \
            self.dfsim['y']**2. + self.dfsim['z']**2.) > 20.]

        elif self.rcut == False and self.SDSS == False:
            self.df = self.dfsim
        elif self.SDSS == True and self.rcut == False:
            self.df = self.dfsim[self.dfsim['sdss'] ==1]
        else:
            self.df = self.dfsim
        return print("dataframes created and Cuts performed")

    def setvars(self):
        vars = ['self.d', 'self.l', 'self.b', 'self.pma', 'self.pmd', 'self.ed',
        'self.epma', 'self.epmd', 'self.vlos', 'self.evlos']
        dats = ['\'dist\'', '\'l\'', '\'b\'', '\'dmu_l\'', '\'dmu_b\'', '\'edist\'',
        '\'edmu_l\'', '\'edmu_b\'', '\'vlos\'', '\'evlost\'']
        for i in range(len(dats)):
            exec("%s = np.array(self.df[%s])"%(vars[i],dats[i]))

        self.N = len(self.df['x'])
        self.l = np.deg2rad(self.l)
        self.b = np.deg2rad(self.b)
        return

    def cpos_icrs(self, l, b, d):
        """
        The cartesian positional coordinates in the ICRS reference frame
        caluclated from the galacitc coordinates l,b and distance.kinematics_mock_2
        """
        c = np.cos
        s = np.sin
        A = np.array([c(b)*c(l), c(b)*s(l),s(b)]).T


        if self.radec == True:
            x,y,z = self.T @ A*d
        else:
            x,y,z = A*d
        return x, y, z

    def cposerr_icrs(self):
        """
        The cartesian positional errors in the ICRS reference frame
        caluclated from the galacitc coordinates l,b and distance and
        their errors. Calculated by monte carlo error propagation.
        """

        def mcerr(l,b,d,ed):

            N = 100
            dp = np.random.normal(d, ed, N)
            xp = np.zeros(N)
            yp = np.zeros(N)
            zp = np.zeros(N)

            for i in range(N):
                # xp[i] = dp[i]*np.cos(l)*np.cos(b)
                # yp[i] = dp[i]*np.sin(l)*np.cos(b)
                # zp[i] = dp[i]*np.sin(b)

                #now taking into account if radec==True
                xp[i], yp[i], zp[i] = self.cpos_icrs(l, b, dp[i])

            ex = np.std(xp)
            ey = np.std(yp)
            ez = np.std(zp)
            return ex, ey, ez

        self.ex = np.zeros(self.N)
        self.ey = np.zeros(self.N)
        self.ez = np.zeros(self.N)

        for i in range(self.N):
            self.ex[i], self.ey[i], self.ez[i] = mcerr(self.l[i], self.b[i], \
            self.d[i], self.ed[i])

        return print("Positional errors computed")

    def get_cpos_icrs(self):

        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        self.z = np.zeros(self.N)

        for i in range(self.N):
            self.x[i], self.y[i], self.z[i] = self.cpos_icrs(self.l[i], self.b[i], self.d[i])

        return print("Cartesian positions computed from galactic coordinates")

    def convert_icrs_mw(self):
        self.x += self.r_sun_mw[0]
        self.y += self.r_sun_mw[1]
        self.z += self.r_sun_mw[2]
        return print("positions converted to mw frame")


    def v_icrs(self, l, b, d, vlos, pma, pmd):
        k= 4.74057
        A = np.array([[np.cos(l)*np.cos(b), -np.sin(l), -np.cos(l)*np.sin(b)],
                      [np.sin(l)*np.cos(b),  np.cos(l), -np.sin(l)*np.sin(b)],
                      [np.sin(b)          ,  0.0      ,  np.cos(b)]])
        B = self.T @ A
        v = np.array([vlos, k*d*pma, k*d*pmd]).reshape(-1,1)
        if self.radec == True:
            uvw = np.dot(B,v)
        else:
            uvw = np.dot(A,v)
        return uvw[0], uvw[1], uvw[2]

    def v_icrs_errs(self):
        """
        Function to monte carlo the velocity errors
        """

        def mcerr(l, b, d, vlos, pma, pmd, ed, evlos, epma, epmd):

            N = 100
            vlosp = np.random.normal(vlos, evlos, N)
            pmap  = np.random.normal(pma, epma, N)
            pmdp  = np.random.normal(pmd, epmd, N)
            dp    = np.random.normal(d, ed, N)

            vxp   = np.zeros(N)
            vyp   = np.zeros(N)
            vzp   = np.zeros(N)

            for i in range(N):
                vxp[i], vyp[i], vzp[i] = self.v_icrs(l, b, dp[i], \
                vlosp[i], pmap[i], pmdp[i])

            evx = np.std(vxp)
            evy = np.std(vyp)
            evz = np.std(vzp)
            return evx, evy, evz

        self.evx = np.zeros(self.N)
        self.evy = np.zeros(self.N)
        self.evz = np.zeros(self.N)

        with alive_bar(len(self.x)) as bar:
            for i in range(self.N):

                self.evx[i], self.evy[i], self.evz[i] = mcerr(self.l[i],\
                self.b[i], self.d[i], self.vlos[i], self.pma[i], self.pmd[i], \
                self.ed[i], self.evlos[i], self.epma[i], self.epmd[i])
                bar()
        return print("velocity errors computed")

    def get_v_icrs(self):
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)
        self.vz = np.zeros(self.N)
        print("calculating velocities")
        with alive_bar(len(self.x)) as bar:
            for i in range(self.N):
                self.vx[i], self.vy[i], self.vz[i] = self.v_icrs(self.l[i], self.b[i],
                self.d[i], self.vlos[i], self.pma[i], self.pmd[i])
                bar()
        return

    def convert_icrs_mw_vel(self):
        self.vx += self.v_sun_mw[0]
        self.vy += self.v_sun_mw[1]
        self.vz += self.v_sun_mw[2]
        return

    def cangmom(self):
        """ Cartesian Angular Momentum Components """
        self.lx = (self.y* self.vz - self.z*self.vy)
        self.ly = (self.z* self.vx - self.x*self.vz)
        self.lz = (self.x* self.vy - self.y*self.vx)
        self.lc = np.array([self.lx, self.ly, self.lz])

        return print("cartesian angular momentum computed in MW frame")

    def cangmom_err(self):

        def mcerr(x, y, z, vx, vy, vz, ex, ey, ez, evx, evy, evz):
            def cartang(x, y, z, vx, vy, vz):
                """ Cartesian Angular Momentum Components """
                lx = (y* vz - z*vy)
                ly = (z* vx - x*vz)
                lz = (x* vy - y*vx)
                return lx, ly, lz

            N  = 100
            xp = np.random.normal(x, ex, N)
            yp = np.random.normal(y, ey, N)
            zp = np.random.normal(z, ez, N)

            vxp = np.random.normal(vx, evx, N)
            vyp = np.random.normal(vy, evy, N)
            vzp = np.random.normal(vz, evz, N)

            lxp = np.zeros(N)
            lyp = np.zeros(N)
            lzp = np.zeros(N)

            for i in range(N):
                lxp[i], lyp[i], lzp[i] = cartang(xp[i], yp[i], zp[i], vxp[i], vyp[i], vzp[i])

            lx  = np.mean(lxp)
            ly  = np.mean(lyp)
            lz  = np.mean(lzp)

            elx = np.std(lxp)
            ely = np.std(lyp)
            elz = np.std(lzp)

            rhoxy = spearmanr(lxp, lyp)[0]
            rhoxz = spearmanr(lxp, lzp)[0]
            rhoyz = spearmanr(lyp, lzp)[0]
            return elx, ely, elz, rhoxy, rhoxz, rhoyz

        self.elx = np.zeros(self.N)
        self.ely = np.zeros(self.N)
        self.elz = np.zeros(self.N)

        self.cxy = np.zeros(self.N)
        self.cxz = np.zeros(self.N)
        self.cyz = np.zeros(self.N)

        with alive_bar(len(self.x)) as bar:

            for i in range(self.N):
                self.elx[i], self.ely[i], self.elz[i], self.cxy[i], self.cxz[i],\
                self.cyz[i] = mcerr(self.x[i], self.y[i], self.z[i],\
                self.vx[i], self.vy[i], self.vz[i], self.ex[i], self.ey[i],\
                self.ez[i], self.evx[i], self.evy[i], self.evz[i])
                bar()

        return print("Angular momenta errors computed")

    def prog_b(self, prog_L):
        print("Inside Angmom coord Function")
        """
        Method: calculate stream coordinates using the input angular momentum
        of the progenitor. By performing an axis-angle rotation between the
        MW disc plane defined by the unit vector L=(0,0,1).

        PARAMETERS:
        prog_L : Galactocentric angular momentum of the progenitor
        1x3 Array. Default set to Sgr plane cut."""

        if self.prog_l is None:
            self.prog_l = self.sgr_l
        #unit vector sgr and MW-
        prog_L = self.prog_l/np.linalg.norm(self.prog_l)
        mw_L  = np.array([0.,0.,1.])

        #axis-angle for frame rotation
        th = np.arccos(np.clip(np.dot(prog_L, mw_L), -1.0, 1.0))
        #normal vector to sgr and mw-z
        n = np.cross(prog_L, mw_L)
        #Rotation matrix from axis angle and norm vector
        R = np.array([[np.cos(th) + n[0]**2.*(1 - np.cos(th)),
                       n[0]*n[1]*(1-np.cos(th)) - n[2]*np.sin(th),
                       n[0]*n[2]*(1-np.cos(th)) + n[1]*np.sin(th)],
                      [n[1]*n[0]*(1-np.cos(th)) + n[2]*np.sin(th),
                       np.cos(th) + n[1]**2.*(1 - np.cos(th)),
                       n[1]*n[2]*(1-np.cos(th)) -n[0]*np.sin(th)],
                      [n[2]*n[0]*(1-np.cos(th)) - n[1]*np.sin(th),
                       n[2]*n[1]*(1-np.cos(th)) + n[0]*np.sin(th),
                       np.cos(th) + n[2]**2.*(1 - np.cos(th))]])
        #now shift position momentum coorinates (x, y, z)
        #such that they are in the new frame
        rs = np.c_[self.x, self.y, self.z]

        #positions momenta in new frame p=prime
        rsp = np.array([np.dot(R, rs[i,:]) for i in range(len(self.x))])
        self.prog_r = rsp
        prog_b = np.array([np.arccos(rsp[i,2]/np.linalg.norm(rsp[i,:])) for i in range(len(self.x))])
        prog_b = 90 - np.rad2deg(prog_b)
        print(prog_b, rsp.shape, max(prog_b), min(prog_b))
        #check that prime coord sgr stars point in lz
        # sgrs = np.where(self.df['sgr'] == 2)[0]
        self.progb = prog_b
        return print("prog_b calculated")

    def sgr_b_majewski(self):
        print("Inside Majewski Function")
        SGR_PHI = (180 + 3.75) * u.degree # Euler angles (from Law & Majewski 2010)
        SGR_THETA = (90 - 13.46) * u.degree
        SGR_PSI = (180 + 14.111534) * u.degree

        # Generate the rotation matrix using the x-convention (see Goldstein)
        D = rotation_matrix(SGR_PHI, "z")
        C = rotation_matrix(SGR_THETA, "x")
        B = rotation_matrix(SGR_PSI, "z")
        A = np.diag([1.,1.,-1.])
        SGR_MATRIX = matrix_product(A, B, C, D)

        @frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
        def galactic_to_sgr():
            """ Compute the transformation matrix from Galactic spherical to
                heliocentric Sgr coordinates.
            """
            return SGR_MATRIX

        @frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
        def sgr_to_galactic():
            """ Compute the transformation matrix from heliocentric Sgr coordinates to
                spherical Galactic.
            """
            return matrix_transpose(SGR_MATRIX)

        #define angles l,b in ICRS coordinates
        gal = coord.SkyCoord(l=np.rad2deg(self.l)*u.degree, b = np.rad2deg(self.b)*u.degree, frame='galactic')
        icrs = gal.transform_to('icrs')
        sgrb = icrs.transform_to(Sagittarius)
        self.sgrb = np.array(sgrb.Beta)

        return print("Sgr Beta Computed with Law & Majewski Angles")

    def save_df(self, fname):
        if self.dataset == "halo":
            dat = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz,
                            self.lx, self.ly, self.lz, self.elx, self.ely, self.elz,
                            self.elx, self.ely, self.elz, self.evx, self.evy, self.evz,
                            self.elx, self.ely, self.elz, self.cxy, self.cxz, self.cyz,
                            np.array(self.df['rapo']), np.array(self.df['sdss'])])

        elif (self.dataset == "halo") & (self.bcut_method == "Majewski"):

            dat = np.array([self.lx, self.ly, self.lz, self.elx, self.ely, self.elz,
                            self.cxy, self.cxz, self.cyz, self.sgrb, np.array(self.df['gse']),
                            np.array(self.df['sdss'])])
            if self.bcut is not None:
                bcut = np.where(np.abs(self.sgrb) < self.bcut)[0]
                dat = dat[:, bcut]
                # fafter = len(np.where(dat[-2, :] == 2)[0])/len(dat[0,:])
                print("Sgr plane cut performed")

        elif (self.dataset == "halo") & (self.bcut_method == "angmom"):
            print("performing plane cut")
            dat = np.array([self.lx, self.ly, self.lz, self.elx, self.ely, self.elz,
                            self.cxy, self.cxz, self.cyz, self.progb])

            if self.bcut is not None:
                bcut = np.where(np.abs(self.progb) < self.bcut)[0]
                dat = dat[:, bcut]
                # fafter = len(np.where(dat[-2, :] == 2)[0])/len(dat[0,:])
                print("Stream plane cut performed")
        else:
            print("no file saved")
        # print("fraction before", fbefore, "fraction after", fafter)
        np.savetxt(fname, dat)
        return print("File saved to %s"%(fname))

import numpy as np
from os.path import exists
from numpy.ctypeslib import ndpointer
from ctypes import *
from subprocess import Popen
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, broyden1, basinhopping, ridder

from units import msol, cm, g, s
from quark_eos import load_eos, load_quark_eos, c, Mb

class RNS:
    def __init__(self, eos, MDIV=101, SDIV=201):
        SMAX = .9999

        so_file = f"spin/spin-{MDIV}-{SDIV}.so"
        if not exists(so_file):
            cmd = f"gcc -fPIC --shared -DMDIV={MDIV} -DSDIV={SDIV} "\
                  f"equil_util.c spin.c nrutil.c -lm -o {so_file}"
            if Popen(cmd.split()).wait() != 0:
                raise Exception("compilation failed")

        rns = cdll.LoadLibrary("./"+so_file)
        self.rns = rns
        rns.set_transition.argtypes = [c_double, c_double]

        self.eos = eos

        rns.sphere.argtypes = [
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                c_int,
                c_char_p,
                c_double,
                c_double,
                c_double,
                c_double,
                c_double,
                c_double,
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                POINTER(c_double)]

        rns.spin.argtypes = [
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                c_int,
                c_char_p,
                c_double,
                c_double,
                c_double,
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                c_int,
                c_double,
                c_double,
                c_int,
                POINTER(c_int),
                c_int,
                c_double,
                POINTER(c_double),
                POINTER(c_double)]

        self.rns.mass_radius.argtypes = [
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                c_int,
                c_char_p,
                c_double,
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                ndpointer(np.float64),
                c_double,
                c_double,
                c_double,
                c_double,
                POINTER(c_double),
                POINTER(c_double),
                POINTER(c_double),
                POINTER(c_double),
                ndpointer(np.float64),
                ndpointer(np.float64),
                POINTER(c_double),
                POINTER(c_double)]

        self.mu = np.concatenate(([0],np.linspace(0,1,MDIV)))
        self.s_gp = np.concatenate(([0],SMAX*np.linspace(0,1,SDIV)))

        self.M = c_double(0)
        self.J = c_double(0)
        self.R = c_double(0)
        self.M0 = c_double(0)
        self.Mp = c_double(0)
        self.r_e = c_double(0)
        self.Omega = c_double(0)
        self.Omega_K = c_double(0)

        self.vp = np.zeros((SDIV+1,))
        self.vm = np.zeros((SDIV+1,))


        self.rho = np.zeros(shape=(SDIV,MDIV))
        self.gama = np.zeros(shape=(SDIV,MDIV))
        self.alpha = np.zeros(shape=(SDIV,MDIV))
        self.omega = np.zeros(shape=(SDIV,MDIV))
        self.energy = np.zeros(shape=(SDIV,MDIV))
        self.enthalpy = np.zeros(shape=(SDIV,MDIV))
        self.pressure = np.zeros(shape=(SDIV,MDIV))
        self.velocity_sq = np.zeros(shape=(SDIV,MDIV))

        self.p_surface = 1.01e-7 * c**-2
        self.e_surface = 7.8e-15
        self.h_min = c**-2

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, eos):
        self._eos = eos
        self.lg_e = np.concatenate(([0],np.log10(eos.e/1e15)))
        self.lg_p = np.concatenate(([0],np.log10(eos.p/1e15/c**2)))
        self.lg_h = np.concatenate(([0],np.log10(eos.h/c**2)))
        self.lg_n0 = np.concatenate(([0],np.log10(eos.n0)))
        self.n_tab = eos.n_tab
        self.h_at_e = interp1d(self.lg_e[1:],self.lg_h[1:])
        self.p_at_e = interp1d(self.lg_e[1:],self.lg_p[1:])
        self.rns.set_transition(eos.start, eos.end)

    @property
    def ec(self):
        return self._ec

    @ec.setter
    def ec(self, ec):
        if ec!=None:
            self._ec = ec
            self.hc = 10**(self.h_at_e(np.log10(ec)))
            self.pc = 10**(self.p_at_e(np.log10(ec)))

    def sphere(self, ec):
        self.ec = ec
        self.rns.sphere(self.s_gp, self.lg_e, self.lg_p, self.lg_h,
                self.lg_n0,self.n_tab, b'tab', 0., self.ec, self.pc,
                self.hc, self.p_surface, self.e_surface, self.rho,
                self.gama, self.alpha, self.omega, self.r_e)
    
    def spin(self,r_ratio,ec=None,cf=1,acc=1e-5,max_n=100,print_dif=0):
        if not .5<r_ratio<1:
            return
        self.ec = ec
        self.r_ratio = r_ratio

        n_it = c_int(0)

        self.rns.spin(self.s_gp, self.mu, self.lg_e, self.lg_p, self.lg_h,
                self.lg_n0, self.n_tab, b'tab', 0., self.hc, self.h_min,
                self.rho, self.gama, self.alpha, self.omega, self.energy,
                self.pressure, self.enthalpy, self.velocity_sq, 0, acc,
                cf, max_n, n_it, print_dif, r_ratio, self.r_e, self.Omega)

        assert n_it.value < max_n, Exception("not converged")

        self.rns.mass_radius(self.s_gp, self.mu, self.lg_e, self.lg_p,
                self.lg_h, self.lg_n0, self.n_tab, b'tab', 0., self.rho,
                self.gama, self.alpha, self.omega, self.energy,
                self.pressure, self.enthalpy, self.velocity_sq,
                self.r_ratio, self.e_surface, self.r_e, self.Omega,
                self.M, self.M0, self.J, self.R, self.vp, self.vm,
                self.Omega_K, self.Mp)

        self.T = .5 * self.Omega.value * self.J.value

    def spin_down(self, de):
        M0 = self.M0.value
        ec = self.ec
        def obj(r_ratio):
            self.spin(r_ratio, print_dif=0)
            print(f"r_ratio={r_ratio}\tM0={self.M0.value}")
            return (self.M0.value-M0)/(msol*c**2)
        undone = True
        while undone:
            self.ec = ec + de
            undone = False
            try: ridder(obj, self.r_ratio-.1, self.r_ratio+.1)
            except ValueError:
                undone = True
                de /= 2.
                assert de > 1e-5, ValueError("Minimal de archieved")
                print(f"de={de}")

get_m2 = lambda B=1e11, R=1e4:\
        (2*np.pi*R**3*B)**2 / (4*np.pi*1e-7) * 1e13 * g*cm**5*s**-2

if __name__ == "__main__":
    eos = load_quark_eos(e0=.3, e1=1)
    #eos = load_eos("eosSLy")
    rns = RNS(eos, 101, 201)
    rns.sphere(1.)
    rns.spin(.8)
    print(f"M0={rns.M0.value}")
    rns.spin_down(.1)
"""
    r_ratio = np.linspace(.999, .6, 20)
    M0 = []
    for _r_ration in r_ratio:
        print(_r_ration)
        rns.spin(_r_ration,print_dif=0)
        M0.append(rns.M0.value)
    #solve = lambda f,x0: fsolve(f,x0,xtol=1e-8,factor=1e10)
    def solve(f,x0):
        def g(x):
            x = f(x)
            return x[0]*x[0] + x[1]*x[1]
        return basinhopping(g, x0, T=.1)
    M0, T = rns.M0.value, rns.T
    for _T in np.linspace(rns.T, rns.T*0.9, 100):
        rns.find(T = _T)
    dt = 1e6
    B = 1e11 # tesla
    m2 = get_m2(B, rns.R.value*1e-2)
    """



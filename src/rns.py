import numpy as np
from units import c
from time import time
from os.path import exists
from numpy.ctypeslib import ndpointer
from ctypes import *
from subprocess import Popen
from scipy.interpolate import interp1d
from scipy.optimize import (fsolve, broyden1, basinhopping, ridder,
        toms748)

class RNS:
    def __init__(self, eos, MDIV=101, SDIV=201, SMAX=.9999, LMAX=10):
        SMAX = .9999

        so_file = f"spin/spin-{MDIV}-{SDIV}-{SMAX}-{LMAX}.so"

        if not exists(so_file):
            cmd = f"gcc -fPIC --shared -DMDIV={MDIV} -DSDIV={SDIV} "\
                  f"-DSMAX={SMAX} -DLMAX={LMAX} "\
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
            try:
                self.hc = 10**(self.h_at_e(np.log10(ec)))
                self.pc = 10**(self.p_at_e(np.log10(ec)))
            except ValueError:
                raise Exception(f"interp err: ec={ec}")

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

        return self

    def spin_down(self, ec, dec=1e-4, M0=None, solve=ridder):
        if M0==None: M0 = self.M0.value
        def obj(r_ratio):
            self.spin(r_ratio)
            #print(f"r_ratio={r_ratio}\tM0={self.M0.value}")
            return (self.M0.value-M0)/M0
        prev = []
        for _ in range(3):
            #print("spin down:",self.ec,self.r_ratio)
            self.ec += dec
            t = time()
            _,msg = solve(obj, self.r_ratio+.01,self.r_ratio-.01,
                    full_output=True)
            t = time()-t
            #print(msg.function_calls, t, (self.M0.value - M0)/M0)
            prev.append(self.r_ratio)
            yield self
        while (self.ec < ec) == (dec > 0):
            #print("spin down:",self.ec,self.r_ratio)
            self.ec += dec
            r_ratio = prev[0] - 3*prev[1] + 3*prev[2]
            try:
                t = time()
                _,msg = solve(obj, r_ratio+5e-4, r_ratio-5e-4, 
                        full_output=True)
                t = time()-t
            except ValueError:
                t = time()
                _,msg = solve(obj, r_ratio+1e-1, r_ratio-1e-1, 
                        full_output=True)
                t = time()-t
            print(msg.function_calls, t, (self.M0.value - M0)/M0)
            print(f"guess: {r_ratio:3.3e}, real: {self.r_ratio:3.3e} "\
                  f"diff: {r_ratio-self.r_ratio:3.3e}")
            prev = prev[1:] + [self.r_ratio]
            yield self


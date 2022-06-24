import numpy as np
from time import time
from units import c,msol,g,cm,s
from ctypes import *
from os.path import exists
from subprocess import Popen
from collections import namedtuple
from scipy.optimize import ridder
from numpy.ctypeslib import ndpointer
from scipy.interpolate import interp1d

from quark_eos import load_eos, load_quark_eos, Mb

get_m2 = lambda B=1e11, R=1e4:\
        (2*np.pi*R**3*B)**2 / (4*np.pi*1e-7) * 1e13 * g*cm**5*s**-2

class RNS:
    def __init__(self, eos, MDIV=101, SDIV=201, SMAX=.9999, LMAX=10):
        SMAX = .9999

        so_file = f"spin/spin-{MDIV}-{SDIV}-{SMAX}-{LMAX}.so"

        if not exists(so_file):
            cmd = f"gcc -fPIC --shared -DMDIV={MDIV} -DSDIV={SDIV} "\
                  f"-DSMAX={SMAX} -DLMAX={LMAX} "\
                  f"equil_util.c spin.c nrutil.c -lm -o {so_file}"
            if Popen(cmd.split()).wait() != 0:
                raise RuntimeError("compilation failed")

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
    def values(self):
        ans = namedtuple("RNS", ["M","M0","r_ratio", "R", "Omega",
            "Omega_K", "J","T", "Mp"])
        return ans(self.M.value, self.M0.value, self.r_ratio, self.R.value,
                self.Omega.value, self.Omega_K.value, self.J.value,
                self.T, self.Mp.value)

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
                raise ValueError(f"interp err: ec={ec}")

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

        assert n_it.value < max_n, "not converged"

        self.rns.mass_radius(self.s_gp, self.mu, self.lg_e, self.lg_p,
                self.lg_h, self.lg_n0, self.n_tab, b'tab', 0., self.rho,
                self.gama, self.alpha, self.omega, self.energy,
                self.pressure, self.enthalpy, self.velocity_sq,
                self.r_ratio, self.e_surface, self.r_e, self.Omega,
                self.M, self.M0, self.J, self.R, self.vp, self.vm,
                self.Omega_K, self.Mp)

        self.T = .5 * self.Omega.value * self.J.value

        return self

    def spin_down(self, ec, dec=1e-2, disp=False, alp=.7):
        M0 = self.M0.value
        obj = lambda x: self.spin(x,acc=1e-7).M0.value / M0 - 1
        prev = []
        last_err = 1e-2
        while (self.ec < ec) == (dec > 0):
            self.ec += dec
            if len(prev) < 3:
                r_ratio = self.r_ratio
            else:
                r_ratio = 3*prev[2] - 3*prev[1] + prev[0]
            delta = last_err
            low, high = r_ratio, r_ratio+delta
            t = time()
            flow, fhigh = obj(low), obj(high)
            function_calls = 2
            skip = False
            while flow*fhigh > 0:
                delta *= 2
                function_calls += 1
                if fhigh>0:
                    low, high = high, high+delta
                    flow,fhigh = fhigh, obj(high)
                else:
                    low, high = low-delta, low
                    flow,fhigh = obj(low), flow
                if high>1:
                    high = .9999
                    fhigh = obj(high)
                    if fhigh > 0:
                        return
            if disp: 
                print("%.5f %.5f %.5f %.5e %.5e" % (
                    low,high,delta,obj(low),obj(high)))
            _, msg = ridder(obj, low, high, xtol=1e-5, full_output=True)
            t = time() - t
            prev.append(self.r_ratio)
            last_err = last_err*(1-alp) + abs(self.r_ratio-r_ratio)*alp
            if len(prev)>3: prev.pop(0)
            if disp:
                print("%d+%d %.2f %.3e %.3e %.3e %.3e" % (
                    function_calls, msg.function_calls, t,
                    (self.M0.value-M0)/M0, r_ratio, self.r_ratio,
                    self.r_ratio - r_ratio))
            yield self

    def is_stable(self, dec=1e-4):
        J = self.J.value
        M = self.M.value
        r_ratio = self.r_ratio
        self.ec += dec
        res, msg = ridder(lambda x:self.spin(x,acc=1e-7).J.value/J-1,
                r_ratio-1e-2, r_ratio+1e-2, full_output=True, xtol=1e-5)
        print(msg)
        stable = self.M.value > M
        self.ec -= dec
        self.spin(r_ratio)
        return stable
        

__all__ = ['get_m2', 'load_eos', 'load_quark_eos', 'c', 'Mb', 'msol',
        'cm', 'g', 'RNS', 'ridder']


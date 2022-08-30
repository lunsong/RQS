import numpy as np
from scipy.interpolate import (interp1d, UnivariateSpline,
        CubicSpline)
from scipy.integrate import solve_ivp, quad

from scipy.optimize import ridder

from os.path import exists

from collections import namedtuple

EOS = namedtuple("EOS",["e","p","h","n0","n_tab","start","end"])

c = 2.99792458e10
Mb=1.66e-24

def load_eos(eos):
    with open(eos,"r") as f:
        x = f.read().strip().split("\n")
    x = list(map(lambda s:list(map(float, s.split())),x))
    if len(x[0])==1:
        n_tab, = x[0]
        start = end = -10
    else:
        n_tab, start, end = x[0]
    x = np.array(x[1:])
    e,p,h,n = x.T
    return EOS(e,p,h,n,int(n_tab),int(start),int(end))

def quark_eos(e0,e1,eos="eosSLy", construction="Maxwell",
        ss=1/np.sqrt(3),Gama=1.03):
    """
    e0,e1 are the start and the end of phase transition, in 1e15
    eos is the base eos, default eosSLy
    mode=write:write new eos
         regenerate:regenerate base eos to validate the algorithm
         otherwise: return the generated eos
    """

    e,p,h,n,_,_,_ = load_eos(eos)
    lge,lgp = map(np.log, (e,p))
    rho = n*Mb

    _p_at_e = CubicSpline(lge,lgp)
    _dp_de = _p_at_e.derivative()
    p_at_e = lambda x:np.exp(_p_at_e(np.log(x)))

    p0 = p_at_e(e0)
    #p1 = p0*(1+1e-8)
    p1 = p0
    if e1==None:
        B = 8.3989e34 / c**2
        e1 = 3*p0/c**2+4*B

    assert e1>e0, ValueError(f"e0={e0:e} should be smaller than e1={e1:e}")

    if construction=="Maxwell":
        def p_at_e(e):
            if e<e0:
                return p_at_e(e)
            if e<e1:
                return p0*(e-e1)/(e0-e1)+p1*(e-e0)/(e1-e0)
            return c**2*(e-e1) * ss**2+p1
        def dp_de(e):
            if e<e0:
                return p_at_e(e) / e * _dp_de(np.log(e)) 
            if e<e1:
                return 0;
            return c**2 * ss**2.

        e = np.concatenate((
            e[e<e0],
            np.linspace(e0,e1,10),
            np.linspace(e1,e[-1],80)[1:]))
    elif construction=="Gibbs":
        A = (e0 - p0/c**2/(Gama-1)) * p0**(-1/Gama)
        poly = lambda p: A*p**(1/Gama)+p/c**2/(Gama-1)
        p1 = ridder(lambda p: poly(p)-e1, p0, p0*10)
        def p_at_e(e):
            if e<e0:
                return p_at_e(e)
            if e<e1:
                return ridder(lambda p: poly(p)-e, p0,p1)
            return c**2*(e-e1) * ss**2 + p1
        def dp_de(e):
            if e<e0:
                return p_at_e(e) / e * _dp_de(np.log(e)) 
            if e<e1:
                p = p_at_e(e)
                return 1/(A/Gama*rho**(1/Gama-1)+1/c**2/(Gama-1))
            return c**2 * ss**2
        e = np.concatenate((
            e[e<e0],
            np.linspace(e0,e1,30),
            np.linspace(e1,e[-1],80)[1:]))

    #enew = e
    p = np.array(list(map(p_at_e, e)))

    rho = solve_ivp(lambda t,y: y / (t+p_at_e(t)/c**2),
            (e[0],e[-1]), (rho[0],), t_eval=e, rtol=3e-14).y[0]

    n = rho / Mb

    h = solve_ivp(lambda t,y:dp_de(t)/(t+p_at_e(t)/c**2),
            (e[0],e[-1]),(h[0],),t_eval=e,rtol=1e-14).y[0]

    return EOS(e,p,h,n,e.shape[0],sum(e<=e0), sum(e<=e1))
    if ofile != None:
        with open(ofile, "w") as f:
            f.write(f"{enew.shape[0]} {sum(enew<=e0)} {sum(enew<=e1)}\n")
            #f.write(f"{enew.shape[0]}\n")
            for e,p,h,n in zip(enew,pnew,hnew,nnew):
                f.write(f"{e:.16e} {p:.16e} {h:.16e} {n:.16e}\n")

def load_quark_eos(e0,e1,base="eosSLy"):
    file = f"quark_eos/{base}-{e0}-{e1}.eos"
    if not exists(file):
        quark_eos(e0*1e15,e1*1e15,ofile=file)
    return load_eos(file)


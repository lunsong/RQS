import numpy as np
from scipy.optimize import ridder
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from collections import namedtuple
EOS = namedtuple("EOS",["e","p","h","n0","n_tab","start","end"])

from os.path import exists

c  = 2.99792458e10
Mb = 1.66e-24

def quark_eos(e0,e1,eos="eosSLy", construction="Maxwell",
        ss=1/np.sqrt(3),Gama=1.03):
    """
    Quark Star EoS, all in cgs unit
    e0,e1 are the starting and ending energy density of the phase
    transition. eos is the hardronic EoS. Other parameters are
    from 1811.10929
    """

    with open(eos,"r") as f: x = f.read().strip().split("\n")
    x = list(map(lambda s:list(map(float, s.split())),x))
    x = np.array(x[1:])
    e,p,h,n = x.T
    lge,lgp = map(np.log, (e,p))
    rho = n*Mb

    _p_at_e = CubicSpline(lge,lgp)
    _dp_de = _p_at_e.derivative()
    p_at_e_old = lambda x:np.exp(_p_at_e(np.log(x)))

    p0 = p_at_e_old(e0)

    assert e1>e0, ValueError("e1<=e0!")

    def p_at_e(e):
        if e<=e0: return p_at_e_old(e)
        if e<e1: return p_at_e_tr(e)
        return c**2*(e-e1) * ss**2 + p1
    def dp_de(e):
        if e<=e0: return p_at_e(e) / e * _dp_de(np.log(e)) 
        if e<e1: return dp_de_tr(e)
        return c**2 * ss**2

    if construction=="Maxwell":
        p1 = p0 * 1.0000001
        p_at_e_tr = lambda e: p0*(e-e1)/(e0-e1)+p1*(e-e0)/(e1-e0)
        dp_de_tr = lambda e: (p1-p0) / (e1-e0)

    elif construction=="Gibbs":
        A = (e0 - p0/c**2/(Gama-1)) * p0**(-1/Gama)
        poly = lambda p: A*p**(1/Gama)+p/c**2/(Gama-1)
        p1 = ridder(lambda p: poly(p)-e1, p0, p0*10)
        p_at_e_tr = lambda e: ridder(lambda p: poly(p)-e, p0,p1)
        dp_de_tr = lambda e: 1/(
                A/Gama * p_at_e_tr(e)**(1/Gama-1) + 1/c**2/(Gama-1) )

    e = np.concatenate((
        e[e<e0],
        np.linspace(e0,e1,10),
        np.linspace(e1,e[-1],80)[1:]))

    p = np.array(list(map(p_at_e, e)))

    rho = solve_ivp(lambda t,y: y / (t+p_at_e(t)/c**2),
            (e[0],e[-1]), (rho[0],), t_eval=e, rtol=3e-14).y[0]

    n = rho / Mb

    h = solve_ivp(lambda t,y:dp_de(t)/(t+p_at_e(t)/c**2),
            (e[0],e[-1]),(h[0],),t_eval=e,rtol=1e-14).y[0]

    return EOS(e,p,h,n,e.shape[0],sum(e<=e0), sum(e<=e1))

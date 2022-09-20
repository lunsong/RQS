import numpy as np
from scipy.optimize import ridder
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from collections import namedtuple
EOS = namedtuple("EOS",["e","p","h","n0","n_tab","start","end","e0","e1"])

from os.path import exists

c  = 2.99792458e10
Mb = 1.66e-24

def quark_eos(e0,e1,e2=None,e_qcd=None,eos="eosSLy",
        cons="Maxwell", ss1=1/np.sqrt(3),ss2=None,Gama=1.03):
    """
    Quark Star EoS, all in cgs unit
    e0,e1 are the starting and ending energy density of the phase
    transition. eos is the hardronic EoS. ss1 is the constant sound
    speed after the transition. ss2 is the sound speed after e_qcd.
    Gama is used in Gibbs construction. e2 is used in mixed construction.
    """

    e0 *= 1e15
    e1 *= 1e15
    if e2 != None: e2 *= 1e15
    if e_qcd != None: e_qcd *= 1e15

    #==================== the skeleton =====================#

    def p_at_e(e):
        if e<=e0: return p_at_e_old(e)
        if e<e1: return p_at_e_tr(e)
        return p_at_e_aft(e)
    def dp_de(e):
        if e<=e0: return p_at_e(e) / e * _dp_de(np.log(e)) 
        if e<e1: return dp_de_tr(e)
        return dp_de_aft(e)

    #==================== hadronic eos ======================#

    with open(eos,"r") as f: x = f.read().strip().split("\n")
    x = list(map(lambda s:list(map(float, s.split())),x))
    x = np.array(x[1:])
    e,p,h,n = x.T
    lge,lgp = map(np.log, (e,p))
    rho = n*Mb

    _p_at_e = CubicSpline(lge,lgp)
    _dp_de = _p_at_e.derivative()
    p_at_e_old = lambda x:np.exp(_p_at_e(np.log(x)))

    #================== phase transition ===================#

    p0 = p_at_e_old(e0)

    assert e1>e0, ValueError("e1<=e0!")

    if cons=="Maxwell":
        p1 = p0
        p_at_e_tr = lambda e: p0
        dp_de_tr = lambda e: 0

    elif cons=="Gibbs":
        A = (e0 - p0/c**2/(Gama-1)) * p0**(-1/Gama)
        poly = lambda p: A*p**(1/Gama)+p/c**2/(Gama-1)
        p1 = ridder(lambda p: poly(p)-e1, p0, p0*10)
        p_at_e_tr = lambda e: ridder(lambda p: poly(p)-e, p0,p1)
        dp_de_tr = lambda e: 1/(
                A/Gama * p_at_e_tr(e)**(1/Gama-1) + 1/c**2/(Gama-1) )

    elif cons=="Maxwell-Gibbs":
        A = (e2 - p0/c**2/(Gama-1)) * p0**(-1/Gama)
        poly = lambda p: A*p**(1/Gama)+p/c**2/(Gama-1)
        p1 = ridder(lambda p: poly(p)-e1, p0, p0*10)
        p_at_e_tr = lambda e:\
                p0 if e < e2 else\
                ridder(lambda p: poly(p)-e, p0,p1)
        dp_de_tr = lambda e:\
                0 if e < e2 else\
                1/( A/Gama * p_at_e_tr(e)**(1/Gama-1) + 1/c**2/(Gama-1) )

    #================ constant sound speed =================#

    if e_qcd == None:
        p_at_e_aft = lambda e: c**2 * (e-e1) * ss1**2 + p1
        dp_de_aft  = lambda e: c**2 * ss1**2
    else:
        assert ss2 != None, "sound speed for quark matter not given"
        p2 = c**2 * (e_qcd-e1) * ss1**2 + p1
        p_at_e_aft = lambda e:\
                c**2 * (e-e1) * ss1**2 + p1 if e < e_qcd else\
                c**2 * (e-e_qcd) * ss2**2 + p2
        dp_de_aft = lambda e:\
                c**2 * ss1**2 if e < e_qcd else c**2 * ss2**2

    #==================== integration ======================#

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
    
    if cons == "Maxwell": _e0 = e0; _e1 = e1
    if cons == "Gibbs": _e0 = _e1 = None
    if cons == "Maxwell-Gibbs": _e0 = e0; _e1 = e2

    return EOS(e,p,h,n,e.shape[0],sum(e<=e0), sum(e<=e1), _e0, _e1)

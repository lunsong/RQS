import numpy as np
from scipy.interpolate import (interp1d, UnivariateSpline,
        CubicSpline)
from scipy.integrate import solve_ivp, quad

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

def quark_eos(e0,e1=None,eos="eosSLy",regenerate=False, ofile=None):
    """
    e0,e1 are the start and the end of phase transition, in 1e15
    eos is the base eos, default eosSLy
    mode=write:write new eos
         regenerate:regenerate base eos to validate the algorithm
         otherwise: return the generated eos
    """

    e,p,h,n,_,_,_ = load_eos(eos)
    lge,lgp,lgh,lgn = map(np.log, (e,p,h,n))
    rho = n*Mb
    lgrho = np.log(rho)

    _rho_at_e = interp1d(lge,lgrho,kind="quadratic")
    rho_at_e = lambda x:np.exp(_rho_at_e(np.log(x)))

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

    if regenerate:
        p_at_e_new = p_at_e
        dp_de = lambda e: p_at_e(e) / e * _dp_de(np.log(e))
        enew = e
    else:
        def p_at_e_new(e):
            if e<e0:
                return p_at_e(e)
            if e<e1:
                return p0*(e-e1)/(e0-e1)+p1*(e-e0)/(e1-e0)
            return c**2*(e-e1)/3+p1
        def dp_de(e):
            if e<e0:
                return p_at_e(e) / e * _dp_de(np.log(e)) 
            if e<e1:
                return 0;
            return c**2 / 3.

        enew = np.concatenate((
            e[e<e0],
            np.linspace(e0,e1,10),
            np.linspace(e1,e[-1],80)[1:]))
    #enew = e
    pnew = np.array(list(map(p_at_e_new, enew)))

    rhonew = solve_ivp(lambda t,y: y / (t+p_at_e_new(t)/c**2),
            (e[0],e[-1]), (rho[0],), t_eval=enew, rtol=3e-14).y[0]

    nnew = rhonew / Mb

    hnew = solve_ivp(lambda t,y:dp_de(t)/(t+p_at_e_new(t)/c**2),
            (enew[0],enew[-1]),(h[0],),t_eval=enew,rtol=1e-14).y[0]

    if regenerate:
        perr = max(abs((pnew-p)/p))
        nerr = max(abs((nnew-n)/n))
        eerr = max(abs((enew-e)/e))
        herr = max(abs((hnew-h)/h))
        print(f"p {perr} n {nerr} e {eerr} h {herr}")
    elif ofile != None:
        with open(ofile, "w") as f:
            f.write(f"{enew.shape[0]} {sum(enew<=e0)} {sum(enew<=e1)}\n")
            #f.write(f"{enew.shape[0]}\n")
            for e,p,h,n in zip(enew,pnew,hnew,nnew):
                f.write(f"{e:.16e} {p:.16e} {h:.16e} {n:.16e}\n")
    else:
        return EOS(enew,pnew,hnew,nnew,enew.shape[0],sum(enew<=e0),
                sum(enew<=e1))

def load_quark_eos(e0,e1,base="eosSLy"):
    file = f"quark_eos/{e0}-{e1}.eos"
    if not exists(file):
        quark_eos(e0*1e15,e1*1e15,ofile=file)
    return load_eos(file)


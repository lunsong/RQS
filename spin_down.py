import numpy as np
from units import cm,g,s
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt

import sys

#data = np.load('quark_out.npy')
with open("quark_series") as f:
    data = f.read()
data = list(map(float, data.strip().split()))
data = np.array(data)
data = data.reshape((data.shape[0]//10, 10))

with open("e0_e1") as f:
    loc = {}
    exec(f.read(),loc)
    e_0 = loc["e0"]
    e_1 = loc["e1"]

r_ratio = data[:,0]
e       = data[:,1]
M       = data[:,2]
M0      = data[:,3]
Re      = data[:,4] # km
Omega   = data[:,5] * s**-1
Omega_K = data[:,6]
I_45    = data[:,7]
J       = data[:,8] * M**2
Mp      = data[:,9]

E = M-Mp
T = .5 * Omega * J
V = E-T

class InterpError(Exception):
    pass

def interpolate(xs,ys,rescale=True):
    """xs.shape=(2,N), ys.shape=(M,N) where N is the number of points"""
    if rescale:
        k0 = 1/(np.max(xs[0])-np.min(xs[0]))
        k1 = 1/(np.max(xs[1])-np.min(xs[1]))
        l0 = -np.min(xs[0]) * k0
        l1 = -np.min(xs[1]) * k1
        xs = (xs[0]*k0+l0, xs[1]*k1+l1)
    tri = Delaunay(np.array(xs).T)
    ys = np.array(ys).T
    def f(a,b):
        x = (k0*a+l0,k1*b+l1)
        idx = tri.find_simplex(x)
        if idx==-1:
            #return np.zeros(ys.shape[1])+np.nan
            raise InterpError(f"out of interp area: {a,b}")
        idx = tri.simplices[idx]
        x1,x2,x3 = tri.points[idx]
        y1,y2,y3 = ys[idx]
        y = np.cross(x-x1,x-x2) / np.cross(x3-x1,x3-x2) * y3
        y += np.cross(x-x2,x-x3) / np.cross(x1-x2,x1-x3) * y1
        y += np.cross(x-x3,x-x1) / np.cross(x2-x3,x2-x1) * y2
        return y
    f.plot = lambda : delaunay_plot_2d(tri).show()
    f.back = lambda a,b: ((a-l0)/k0,(b-l1)/k1)
    return f

M_at_e_j = interpolate((e,J),(M,))
Omega_e_at_M0_E_0 = interpolate((M0[e<=e_0],E[e<=e_0]),
        (Omega[e<=e_0],e[e<=e_0]))
Omega_e_at_M0_J_0 = interpolate((M0[e<=e_0],J[e<=e_0]),
        (Omega[e<=e_0],e[e<=e_0]))
Omega_e_at_M0_E_1 = interpolate((M0[e>=e_1],E[e>=e_1]),
        (Omega[e>=e_1],e[e>=e_1]))
Omega_e_at_M0_J_1 = interpolate((M0[e>=e_1],J[e>=e_1]),
        (Omega[e>=e_1],e[e>=e_1]))

get_m2 = lambda B=1e11, R=1e4:\
        (2*np.pi*R**3*B)**2 / (4*np.pi*1e-7) * 1e13 * g*cm**5*s**-2
get_m2.__doc__ = """
B is the surface magnetic field, R is the radius, both in SI units
Returns the magnetic moment in geometric units"""

def evolve(M0=2,E=None,dt=1e6,m2=get_m2(),N=10000):
    """params are in geometric units"""
    #if T==None: T = max_T(M0)
    t = dt
    #J_quark = False
    quark = False
    try:
        Omega, e = Omega_e_at_M0_E_0(M0,E)
    except InterpError:
        #J_quark = T_quark = True
        quark = True
        Omega, e = Omega_e_at_M0_E_1(M0,E)
    #J = 2*T/Omega
    try:
        for step in range(N):
            #breakpoint()
            if step%100==0:print(f"{Omega}{' '*10}{T}{' '*10}", end="\n")
            #if J_quark:
            #    Omega,e0 = Omega_e_at_M0_J_1(M0,J)
            #else:
            #    try:
            #        Omega,e0 = Omega_e_at_M0_J_0(M0,J)
            #    except InterpError:
            #        J_quark = True
            #        Omega,e0 = Omega_e_at_M0_J_1(M0,J)
            #dj = m2 * Omega**3 / (6*np.pi)
            #dm_j = dj * Omega
            #J -= dt * dj
            if quark:
                Omega,e = Omega_e_at_M0_T_1(M0,T)
            else:
                try:
                    Omega,e = Omega_e_at_M0_T_0(M0,T)
                except InterpError:
                    quark = True
                    Omega,e = Omega_e_at_M0_T_1(M0,T)
            dm = m2 * Omega**4 / (6*np.pi)
            E -= dt * dm
            yield t,dm,e
            t += dt
    except InterpError:
        print("run out of interp area")
        pass


fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

def plot_res(M0,res):
    res = np.array(res)
    res[:,0] /= s
    res[:,1:3] /= g*cm**2*s**-3
    ax1.loglog(res[:,0],res[:,1],label=f"T,$M_0={M0}$")
    ax1.loglog(res[:,0],res[:,2],label=f"J,$M_0={M0}$")
    ax2.semilogx(res[:,0],res[:,3],label=f"T,$M_0={M0}$")
    ax2.semilogx(res[:,0],res[:,4],label=f"J,$M_0={M0}$")

def run(M0,T=None):
    res = list(evolve(M0,T=T))
    plot_res(M0,res)

def show():
    ax1.legend(fontsize="xx-small")
    plt.xlabel("time / s")
    ax1.set_ylabel("power / $erg\cdot s^2$")
    ax2.set_ylabel("e / $10^{15} g/cm^3$")
    ax1.grid()
    ax2.grid()
    fig.set_size_inches(6,9)
    fig.show()
    fig.savefig("res",dpi=500)

def max_T(M0):
    def f(T):
        ans = 0
        try:
            Omega_e_at_M0_T_0(M0,T)
        except InterpError:
            ans += 1
        try:
            Omega_e_at_M0_T_1(M0,T)
        except InterpError:
            ans += 1
        return ans==2
    T0=0.005
    T1=0.03
    while T1-T0>.005:
        T2 = (T0+T1)/2
        if f(T2):
            T1 = T2
        else:
            T0 = T2
    return T0-.005

if __name__ == '__main__':
    run(.206,6.337e-5)
    show()
"""
    run(1.748)
    run(1.493)
    run(1.034)
    res = np.array(list(evolve(1.895424,.0311)))
    res[:,0] /= s
    res[:,1] /= g * cm**2 * s**-3
    res[:,2] /= g * cm**2 * s**-3
    ax1.loglog(res[:,0],res[:,1],label="T,$M_0=1.8954$")
    ax1.loglog(res[:,0],res[:,2],label="J,$M_0=1.8954$")
    ax2.semilogx(res[:,0],res[:,3],label="T,$M_0=1.8954$")
    ax2.semilogx(res[:,0],res[:,4],label="J,$M_0=1.8954$")
    res = np.array(list(evolve(2.1808,.0448,.01732)))
    res[:,0] /= s
    res[:,1] /= g * cm**2 * s**-3
    res[:,2] /= g * cm**2 * s**-3
    ax1.loglog(res[:,0],res[:,1],label="T,$M_0=2.1808$")
    ax1.loglog(res[:,0],res[:,2],label="J,$M_0=2.1808$")
    ax2.semilogx(res[:,0],res[:,3],label="T,$M_0=2.1808$")
    ax2.semilogx(res[:,0],res[:,4],label="J,$M_0=2.1808$")
    res = np.array(list(evolve(2.0,.04)))
    res[:,0] /= s
    res[:,1] /= g * cm**2 * s**-3
    res[:,2] /= g * cm**2 * s**-3
    ax1.loglog(res[:,0],res[:,1],label="T,$M_0=2.00$")
    ax1.loglog(res[:,0],res[:,2],label="J,$M_0=2.00$")
    ax2.semilogx(res[:,0],res[:,3],label="T,$M_0=2$")
    ax2.semilogx(res[:,0],res[:,4],label="J,$M_0=2$")
    res = np.array(list(evolve(1.68,.023)))
    res[:,0] /= s
    res[:,1] /= g * cm**2 * s**-3
    res[:,2] /= g * cm**2 * s**-3
    ax1.loglog(res[:,0],res[:,1],label="T,$M_0=1.68$")
    ax1.loglog(res[:,0],res[:,2],label="J,$M_0=1.68$")
    ax2.semilogx(res[:,0],res[:,3],label="T,$M_0=1.68$")
    ax2.semilogx(res[:,0],res[:,4],label="J,$M_0=1.68$")
"""

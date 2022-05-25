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
data = data.reshape((data.shape[0]//9, 9))

with open("e0_e1") as f:
    e0_e1 = {}
    exec(f.read(),locals=e0_e1)

M = data[:,2]
M0 = data[:,3]
e = data[:,1]
J = data[:,-1] * data[:,2]**2
Omega = data[:,5] * s**-1
T = .5 * Omega * J

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
Omega_e_at_M0_T = interpolate((M0,T),(Omega,e))
Omega_e_at_M0_J = interpolate((M0,J),(Omega,e))

get_m2 = lambda B=1e11, R=1e4:\
        (2*np.pi*R**3*B)**2 / (4*np.pi*1e-7) * 1e13 * g*cm**5*s**-2
get_m2.__doc__ = """
B is the surface magnetic field, R is the radius, both in SI units
Returns the magnetic moment in geometric units"""

def evolve(M0=2,T=0.04,dt=1e6,m2=get_m2(),tfinal_s=5e4,N=10000):
    """params are in geometric units"""
    t = dt
    tfinal = tfinal_s * s
    Omega, e = Omega_e_at_M0_T(M0,T)
    J = 2*T/Omega
    try:
        for _ in range(N):
            print(f"{Omega}          {T}  ", end="\n")
            Omega, e0 = Omega_e_at_M0_J(M0,J)
            if np.isnan(Omega):
                print("M0 and J out of interpolation area")
                break
            dj = m2 * Omega**3 / (6*np.pi)
            dm_j = dj * Omega
            J -= dt * dj
            Omega, e1 = Omega_e_at_M0_T(M0,T)
            if np.isnan(Omega):
                print("M0 and T out of interpolation area")
                break
            dm = m2 * Omega**4 / (6*np.pi)
            T -= dt * dm
            yield t,dm,dm_j,e1,e0
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

def run(M0,T):
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


if __name__ == '__main__':
    run(1.748,0.0274)
    run(1.493,0.0183)
    run(1.034,0.0078)
    show()
"""
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

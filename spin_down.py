import numpy as np
from units import cm,g,s
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt

with open("e0_e1") as f:
    loc = {}
    exec(f.read(),loc)
    e_0 = loc["e0"]
    e_1 = loc["e1"]

with open("quark_series") as f:
    data = f.read()
data = list(map(float, data.strip().split()))
data = np.array(data)
data = data.reshape((data.shape[0]//10, 10))

ratio = data[:,0]
e     = data[:,1]
mask  = np.concatenate(((True,), abs(ratio[:-1] - ratio[1:])>1e-2))
for _e in set(e):
    min_r = min(ratio[e==_e])
    mask &= ratio != min_r

mask &= ~( (e>=e_1)&(e<=1.08) )

data = data[mask]

ratio   = data[:,0]
e       = data[:,1]
M       = data[:,2]
M0      = data[:,3]
Re      = data[:,4] # km
Omega   = data[:,5] * s**-1
Omega_K = data[:,6]
I_45    = data[:,7]
J       = data[:,8] * M**2
Mp      = data[:,9]

del data

E = M-Mp
T = .5 * Omega * J
V = E-T

def disp(x,y):
    for _e in e:
        color = "b" if _e <= e_0 else "g"
        plt.plot(x[e==_e],y[e==_e],color)
    plt.show()

class InterpError(Exception):
    def __init__(self, msg, a=None,b=None,x=None):
        super().__init__(msg)
        self.a = a
        self.b = b
        self.x = x

def interpolate(xs,ys,mask=None,rescale=True):
    """xs.shape=(2,N), ys.shape=(M,N) where N is the number of points"""
    if rescale:
        k0 = 1/(np.max(xs[0])-np.min(xs[0]))
        k1 = 1/(np.max(xs[1])-np.min(xs[1]))
        l0 = -np.min(xs[0]) * k0
        l1 = -np.min(xs[1]) * k1
        xs = (xs[0]*k0+l0, xs[1]*k1+l1)
    xs = np.array(xs).T
    ys = np.array(ys).T
    if not mask is None:
        xs = xs[mask]
        ys = ys[mask]
    tri = Delaunay(xs)
    def f(a,b,throw=False):
        x = (k0*a+l0,k1*b+l1)
        idx = tri.find_simplex(x)
        if idx==-1:
            if throw:
                raise InterpError(f"out of interp area: {a,b}", a,b,x)
            return np.zeros(ys.shape[1])+np.nan
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


#B is the surface magnetic field, R is the radius, both in SI units
#Returns the magnetic moment in geometric units
get_m2 = lambda B=1e11, R=1e4:\
        (2*np.pi*R**3*B)**2 / (4*np.pi*1e-7) * 1e13 * g*cm**5*s**-2


Omega_e_at_M0_T_0 = interpolate((M0,T), (Omega,e), e<=e_0)
Omega_e_at_M0_T_1 = interpolate((M0,T), (Omega,e), e>=e_0)
J_V_at_M0_T_0 = interpolate((M0,T), (J,V), e<=e_0)
T_V_at_M0_J_1 = interpolate((M0,J), (T,V), e>=e_1)

def evolve(M0,T,dt=1e6,m2=get_m2(),N=10000):
    """params are in geometric units"""
    t = dt
    quark = False
    Omega, e = Omega_e_at_M0_T_0(M0,T)
    if np.isnan(e) or e>e_0:
        quark = True
        Omega, e = Omega_e_at_M0_T_1(M0,T)
        if np.isnan(e) or e<e_1:
            raise InterpError("case 1")
    for step in range(N):
        if step%1==0:print(f"{Omega:e} {e:e} {T:e}", end="\n")
        if not quark:
            Omega,e = Omega_e_at_M0_T_0(M0,T)
            if np.isnan(e) or e>e_0:
                quark = True
                T += dt*dm
                J,V0 = J_V_at_M0_T_0(M0,T)
                T,V1 = T_V_at_M0_J_1(M0,J)
                print(f"phase transition: delta V={V1-V0}")
        if quark:
            Omega,e = Omega_e_at_M0_T_1(M0,T)
        dm = m2 * Omega**4 / (6*np.pi)
        T -= dt * dm
        yield t,dm,e
        t += dt
    print("run out of interp area")
    pass


#fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

def plot_res(M0,res):
    res = np.array(res)
    res[:,0] /= s
    res[:,1] /= g*cm**2*s**-3
    ax1.loglog(res[:,0],res[:,1],label=f"$M_0={M0}$")
    ax2.semilogx(res[:,0],res[:,2],label=f"$M_0={M0}$")

def run(M0,*args,**kwargs):
    res = list(evolve(M0,*args,**kwargs))
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

"""
if __name__ == '__main__':
    run(.203,5e-5)
    show()
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

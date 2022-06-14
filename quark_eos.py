import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad

linspace = lambda a,b,dx:(i*dx+a for i in range(int((b-a)/dx)))
def cumsum(x):
    ans = 0.0
    for xx in x:
        ans += xx
        yield ans


c = 2.99792458e10
Mb=1.66e-24

eos = "eosSLy"

with open(eos,"r") as f:
    x = f.read().strip().split("\n")
x = list(map(lambda s:list(map(lambda t:float(t),s.split())),x))[1:]
x = np.array(x)
e,p,h,n = x.T
lge,lgp,lgh,lgn = np.log(x.T)
rho = n*Mb
lgrho = np.log(rho)

_rho_at_e = interp1d(lge,lgrho,kind="quadratic")
rho_at_e = lambda x:np.exp(_rho_at_e(np.log(x)))

_p_at_e = interp1d(lge,lgp,kind="quadratic")
p_at_e = lambda x:np.exp(_p_at_e(np.log(x)))

e0 = 0.3e15
p0 = p_at_e(e0)
p1 = p0*(1+1e-8)
B = 8.3989e34 / c**2
#e1 = 3*p0/c**2+4*B
e1 = 1.5e15

assert e1>e0, ValueError(f"e0={e0:e} should be smaller than e1={e1:e}")

print(f"e0={e0:e} e1={e1:e}")
with open("e0_e1","w") as f:
    f.write(f"e0={e0/1e15}\ne1={e1/1e15}")

regenerate = False

if regenerate:
    p_at_e_new = p_at_e
    enew = e
else:
    def p_at_e_new(e):
        if e<e0:
            return p_at_e(e)
        if e<e1:
            return p0*(e-e1)/(e0-e1)+p1*(e-e0)/(e1-e0)
        return c**2*(e-e1)/3+p1
    enew = np.concatenate((
        e[e<e0],
        np.linspace(e0,e1,10),
        np.linspace(e1,e[-1],80)[1:]))
#enew = e
pnew = np.array(list(map(p_at_e_new, enew)))

eps = (e/rho - 1)*c**2
rhonew = solve_ivp(lambda t,y: y / (t+p_at_e_new(t)/c**2),
        (e[0],e[-1]), (rho[0],), t_eval=enew, rtol=3e-14).y[0]

nnew = rhonew / Mb

enew_at_pnew = interp1d(np.log(pnew[(enew<e0)|(enew>=e1)])
    ,np.log(enew[(enew<e0)|(enew>=e1)]),kind="quadratic")#,
    #fill_value="extrapolate")
#_log_pnew = np.linspace(np.log(pnew[0]),np.log(pnew[-1]),100000)
#_pnew = np.exp(_log_pnew)
#_pnew = np.linspace(pnew[0],pnew[-1],100000)
#_log_pnew=np.log(_pnew)
#_enew = np.exp(enew_at_pnew(_log_pnew))
#_hnew = np.cumsum(1 /(_enew+_pnew/c**2)) *(_pnew[1]-_pnew[0])
#_pnew = np.concatenate((pnew[pnew<p0],(p0,),pnew[pnew>p0]))
#hnew = solve_ivp(lambda t,y:1/(np.exp(enew_at_pnew(np.log(t)))+t/c**2),
#        (pnew[0],pnew[-1]),(h[0],),t_eval=_pnew,rtol=1e-14).y[0]
hnew = solve_ivp(lambda t,y:1/(np.exp(enew_at_pnew(np.log(t)))+t/c**2),
        (pnew[0],pnew[-1]),(h[0],),t_eval=pnew,rtol=1e-14).y[0]
#hnew = np.concatenate((hnew[_pnew<p0],
#    np.zeros(sum(pnew==p0))+hnew[_pnew==p0],hnew[_pnew>p0]))
#_hnew_at_enew = interp1d(np.log(_enew),np.log(_hnew),kind="quadratic")
#hnew = np.exp(_hnew_at_enew(np.log(enew)))
#hnew += h[0] - hnew[0]

if __name__ == "__main__":
    if regenerate:
        perr = max(abs((pnew-p)/p))
        nerr = max(abs((nnew-n)/n))
        eerr = max(abs((enew-e)/e))
        herr = max(abs((hnew-h)/h))
        print(f"p {perr} n {nerr} e {eerr} h {herr}")
    else:
        '''
        pnew = pnew[(enew<e0) | (enew>e1)]
        hnew = hnew[(enew<e0) | (enew>e1)]
        nnew = nnew[(enew<e0) | (enew>e1)]
        enew = enew[(enew<e0) | (enew>e1)]
        '''
        with open("quark_eos", "w") as f:
            f.write(f"{enew.shape[0]} {sum(enew<=e0)} {sum(enew<=e1)}\n")
            #f.write(f"{enew.shape[0]}\n")
            for e,p,h,n in zip(enew,pnew,hnew,nnew):
                f.write(f"{e:.16e} {p:.16e} {h:.16e} {n:.16e}\n")

#p_at_rho

e = enew
p = pnew
h = hnew
n0 = nnew

"""
p_at_n = interp1d(lgn,lgp,kind="quadratic")
e_at_n = interp1d(lgn,lge,kind="quadratic")h

dx = 1e-4
lgn_sample = np.mgrid[60:90:dx]
_y = np.exp(e_at_n(lgn_sample))
y = (np.cumsum(np.exp(p_at_n(lgn_sample)-lgn_sample))*dx/c**2+Mb)*np.exp(
        lgn_sample)


dx = 1e-3
N = int((lgp[-1] - lgp[0])/dx/100)
e_at_p = interp1d(lgp,lge,kind="quadratic")
h_at_p = interp1d(lgp,lgh,kind="quadratic")
n_at_p = interp1d(lgp,lgn,kind="quadratic")
_lgp = np.mgrid[lgp[20]+dx:lgp[-1]-dx:dx]
#_lgp = linspace(lgp[0]+dx,lgp[-1]-dx,dx)
_p = np.exp(_lgp)
#_p = map(np.exp, _lgp)
_e = np.exp(e_at_p(_lgp))
_n = np.exp(n_at_p(_lgp))
_h0 = np.exp(h_at_p(_lgp))
#_e = map(lambda x:np.exp(e_at_p(x)), _lgp)
#_h1 = np.cumsum(_p/(_e-_n*1.66e-24+_p/c**2))*dx
_h1 = np.cumsum(_p/(_e+_p/c**2))*dx
#y = cumsum(p * dx / (e + p/c**2) for e,p in zip(_e,_p))
#y = [(_x,_y,np.exp(h_at_p(_x)))
#        for i,(_y,_x) in enumerate(zip(y,_lgp)) if i%N==0]
#y = np.array(y)
#_y = np.exp(h_at_p(_lgp))
"""

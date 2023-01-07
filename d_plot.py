from scipy.integrate import odeint
from scipy.signal import argrelextrema
import numpy as np
from numpy import sin, cos, sqrt
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from itertools import count

_omega = 1.4005
_ma = 1335.535
_f = 6250
_K = 10000
_a = 0.5
_lamda = 656.3616
_lamdat = 151.4388
_Ia = 6779.315
_L = 1690


def get_args(K=_K, Kt=_K, a=_a, pro_id=1, const_K=True):
    if pro_id == 1:
        return (K, -1, a, _omega, _ma, _f, _lamda, -1, -1, -1, const_K)
    elif pro_id == 2:
        return (K, -1, a, 2.2143, 1165.992, 4890, 167.8395, -1, -1, -1, const_K)
    elif pro_id == 3:
        return (K, Kt, a, 1.7152, 1028.876, 3640, 683.4558, 654.3383, 7001.914, 1690, const_K)
    elif pro_id == 4:
        return (K, Kt, a, 1.9806, 1091.099, 1760, 1655.909, 528.5018, 7142.493, 2140, const_K)


def Model1(y, t, K=_K, Kt=_K, a=_a, omega=_omega, ma=_ma, f=_f, lamda=_lamda, lamdat=_lamda, Ia=_Ia, L=_L, const_K=True, rho=1025, g=9.8, mf=4866, mb=2433, k=80000, kt=250000, l=0.5, ht=0.8, hc=3, r=1):
    y1, y2, z1, z2 = y

    dy1 = y2
    dz1 = z2

    Ff = 0
    if np.abs(y1) <= ht:
        Ff = -1/3*rho*g*np.pi*r*r*y1
    elif ht+hc >= np.abs(y1) > ht:
        Ff = rho*g*((np.pi*(r*r)*ht)/3-np.pi*r*r*(y1+ht))
    elif np.abs(y1) >= ht+hc:
        Ff = rho*g*(1/3*np.pi*r*r*ht+np.pi*r*r*hc)

    if not const_K:
        K = K*np.power(np.abs(z2-y2), a)

    dy2 = (f*cos(omega*t)-(mf)*g+k*(z1-y1-ht-l)+K*(z2-y2)-lamda*y2+Ff)/(ma+mf)
    dz2 = (-mb*g-k*(z1-y1-ht-l)-K*(z2-y2))/mb

    return [dy1, dy2, dz1, dz2]


y0 = [-2.80001093, 0, -1.79805351, 0]
y0t = [-2.80001093, 0, -1.79805351, 0, 0, 0, 0, 0]


def Model2(y, t, K=_K, Kt=_K, a=_a, omega=_omega, ma=_ma, f=_f, lamda=_lamda, lamdat=_lamda, Ia=_Ia, L=_L, const_K=True, rho=1025, g=9.8, mf=4866, mb=2433, k=80000, kt=250000, l=0.5, ht=0.8, hc=3, r=1):
    xcf = (3/2*hc**2-1/3*ht**2)/(3*hc+ht)
    mc = (3*hc)/(3*hc+ht)*mf
    mt = (ht)/(3*hc+ht)*mf
    If = 1/12*mc*(3*r**2+hc**2)+mc*(1/2*hc-xcf)**2+3 / \
        20*mt*(r**2+ht**2/4)+mt*(1/3*ht+xcf)**2

    y1, y2, z1, z2, N1, N2, M1, M2 = y

    Ff = 0
    if np.abs(y1) <= ht:
        Ff = -1/3*rho*g*np.pi*r*r*y1
    elif ht+hc >= np.abs(y1) > ht:
        Ff = rho*g*((np.pi*(r*r)*ht)/3-np.pi*r*r*(y1+ht))
    elif np.abs(y1) >= ht+hc:
        Ff = rho*g*(1/3*np.pi*r*r*ht+np.pi*r*r*hc)

    if not const_K:
        K = K*np.power(np.abs(z2-y2), a)

    dy1 = y2
    dz1 = z2

    dy2 = (f*cos(omega*t)-(mf)*g+k*(z1-y1-ht-l)+K*(z2-y2)-lamda*y2+Ff)/(ma+mf)
    dz2 = (-mb*g-k*(z1-y1-ht-l)-K*(z2-y2))/mb

    dN1 = N2
    dM1 = M2

    # dN2 = (Kt*(M2-N2)+kt*(M1-N1))/(mb*(xcf**2+(z1-y1-ht)**2)-2*xcf*(z1-y1-ht)*sin(N1))

    dN2 = 0
    dM2 = 0
    dN2 = (Kt*(M2-N2)+kt*(M1-N1))/(mb*((xcf)**2))
    dM2 = (L*cos(omega*t)-Kt*(M2-N2)-kt*(M1-N1)-lamdat*M2)/(If+Ia)

    return [dy1, dy2, dz1, dz2, dN1, dN2, dM1, dM2]


def get_Es(xf, xb, vf, vb, K=_K, a=_a, const_K=True):
    dv = np.abs(vf-vb)
    dv = (dv[:-1]+dv[1:])/2
    dx = xf-xb
    ddx = np.abs(dx[1:]-dx[:-1])
    if not const_K:
        K = K*np.power(dv, a)
    Es = K*dv*ddx
    return Es

# def get_W(Model, y0, t, args=get_args()):
#     tend = t[-1]
#     nstep = len(t)
#     Tstep = int(nstep/tend*2*np.pi/args[3])
#     sol = odeint(Model, y0, t,args=args)
#     xf = sol[:,0]
#     xb = sol[:,2]
#     vf = sol[:,1]
#     vb = sol[:,3]
#     Es = get_Es(xf,xb,vf,vb,args[0])

#     mean_step = Tstep*10
#     if(nstep<mean_step):
#         return np.sum(Es)/t[-1]
#     else:
#         return np.sum(Es[-Tstep*10:-1])/(t[-1]-t[-Tstep*10])


def solve(Model, y0, t0=100, args=get_args(), log=False, valid=[1, 1]):
    bg = 0
    ed = t0
    delta_t = ed-bg
    t_step = 10
    nstep = delta_t*t_step
    t = np.linspace(bg, ed, nstep)
    Tstep = int(t_step*2*np.pi/args[3])

    nT = 10

    while True:
        sol = odeint(Model, y0, t, args=args)
        Wsum = 0
        flag = True

        if sol.shape[1] >= 4 and valid[0]:
            xf = sol[:, 0]
            xb = sol[:, 2]
            vf = sol[:, 1]
            vb = sol[:, 3]
            Es = get_Es(xf, xb, vf, vb, args[0], args[2], args[-1])

            # 法1
            # W1 = np.sum(Es[int(-3/2*nT*Tstep):int(-1/2*nT*Tstep)])/(t[int(-1/2*nT*Tstep)]-t[int(-3/2*nT*Tstep)])
            # W2 = np.sum(Es[int(-nT*Tstep):-1])/(t[-1]-t[int(-nT*Tstep)])

            # 法2
            Ws = np.cumsum(Es)/(t[1:]-t[0])
            W1 = Ws[int(-3/2*nT*Tstep)]
            W2 = Ws[int(-nT*Tstep)]

            if(np.abs(W1-W2)/W2 < 0.001):
                Wsum += Ws[-1]
            else:
                flag = False

            if log:
                print(ed, W1, W2, Ws[-1])

        if flag and sol.shape[1] >= 8 and valid[1]:
            xf = sol[:, 4]
            xb = sol[:, 6]
            vf = sol[:, 5]
            vb = sol[:, 7]
            Es = get_Es(xf, xb, vf, vb, args[1], args[2], args[-1])

            # 法1
            # W1 = np.sum(Es[int(-3/2*nT*Tstep):int(-1/2*nT*Tstep)])/(t[int(-1/2*nT*Tstep)]-t[int(-3/2*nT*Tstep)])
            # W2 = np.sum(Es[int(-nT*Tstep):-1])/(t[-1]-t[int(-nT*Tstep)])

            # 法2
            Ws = np.cumsum(Es)/(t[1:]-t[0])
            W1 = Ws[int(-3/2*nT*Tstep)]
            W2 = Ws[int(-nT*Tstep)]

            # print(ed,W1,W2)
            if(np.abs(W1-W2)/W2 < 0.001):
                Wsum += Ws[-1]
            else:
                flag = False

            if log:
                print(ed, W1, W2, Ws[-1])

        if flag:
            return Wsum

        bg = ed
        ed *= 2

        delta_t = ed-bg
        nstep = delta_t*50
        Tstep = int(nstep/delta_t*2*np.pi/args[3])
        t = np.linspace(bg, ed, nstep)
        y0 = sol[-1]


def plotModel(Model, y0, tend, args=get_args(), plot_Ws=True):
    nstep = int(tend*20)
    t = np.linspace(0, tend, nstep)
    Tstep = int(nstep/tend*2*np.pi/args[3])
    sol = odeint(Model, y0, t, args=args)
    fig, ax = plt.subplots(3, 2, figsize=(20, 15))

    xf = sol[:, 0]
    xb = sol[:, 2]
    vf = sol[:, 1]
    vb = sol[:, 3]
    # arg = argrelextrema(xf, np.greater)
    # print(np.diff(t[arg]),np.diff(arg))

    # print(sol[-1,0],sol[-1,2])

    i = 0
    ax[0, i].plot(t, sol[:, 0], 'b', label='y1:x_f')
    ax[0, i].plot(t, sol[:, 2], 'g', label='z1:x_b')
    # ax[0].plot(t, sol[:, 0]-sol[:, 2], label='y1-z1:x_f-x_b')

    ax[1, i].plot(t, sol[:, 1], 'b', label='y2:v_f')
    ax[1, i].plot(t, sol[:, 3], 'g', label='z2:v_b')
    # ax[1].plot(t, sol[:, 1]-sol[:, 3], label='y2-z2:v_f-v_b')

    if plot_Ws:
        Es = get_Es(xf, xb, vf, vb, args[0], args[2], args[-1])

        ax[2, i].plot(t[1:], Es, label='Es')

        # cal_t = int(50*Tstep)
        # print(cal_t,t[cal_t])
        # Ws = np.convolve(Es, np.ones(cal_t), 'valid')/(tend/nstep*cal_t)
        # ax[2].plot(t[cal_t:], Ws, label='Ws')

        Ws = np.cumsum(Es)/t[1:]
        ax[2, i].plot(t[1:], Ws, label='Ws')
        ax[2, i].axhline(y=Ws[-1], color='r', linestyle='-')

        print('W =', Ws[-1])

    if sol.shape[1] == 8:
        xf = sol[:, 4]
        xb = sol[:, 6]
        vf = sol[:, 5]
        vb = sol[:, 7]
        # arg = argrelextrema(xf, np.greater)
        # print(np.diff(t[arg]),np.diff(arg))

        # print(sol[-1,0],sol[-1,2])

        i = 1
        ax[0, i].plot(t, xf, 'b', label='y1:x_f')
        ax[0, i].plot(t, xb, 'g', label='z1:x_b')
        # ax[0].plot(t, sol[:, 0]-sol[:, 2], label='y1-z1:x_f-x_b')

        ax[1, i].plot(t, vf, 'b', label='y2:v_f')
        ax[1, i].plot(t, vb, 'g', label='z2:v_b')
        # ax[1].plot(t, sol[:, 1]-sol[:, 3], label='y2-z2:v_f-v_b')

        if plot_Ws:
            Es = get_Es(xf, xb, vf, vb, args[1], args[2], args[-1])

            ax[2, i].plot(t[1:], Es, label='Es')

            # cal_t = int(50*Tstep)
            # print(cal_t,t[cal_t])
            # Ws = np.convolve(Es, np.ones(cal_t), 'valid')/(tend/nstep*cal_t)
            # ax[2].plot(t[cal_t:], Ws, label='Ws')

            Ws = np.cumsum(Es)/t[1:]
            ax[2, i].plot(t[1:], Ws, label='Ws')

            ax[2, i].axhline(y=Ws[-1], color='r', linestyle='-')
            print('Wt =', Ws[-1])

    plt.tight_layout()


n_a = n_K = 30
a_list = np.linspace(0, 1, n_a)
K_list = np.linspace(1, 100000, n_K)
aa, KK = np.meshgrid(a_list, K_list)
# y = np.zeros(n_a*n_K)
yy = []
for a, K in tqdm(zip(aa.flatten(), KK.flatten())):
    if K>100000*a:
        yy.append(solve(Model1, y0, args=get_args(
            K=K, a=a, pro_id=2, const_K=False)))
    else:
        yy.append(0)
yy = np.array(yy).reshape(n_a, n_K)
fig = plt.figure(1, figsize=(12, 8))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_top_view()

ax.plot_surface(aa, KK, yy, rstride=1, cstride=1, cmap='rainbow')
plt.show()


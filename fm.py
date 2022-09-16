from scipy.integrate import odeint
import numpy as np
from numpy import sin, cos, sqrt
import matplotlib.pyplot as plt

def Model1(y, t, omega=1.4005, K=10000, rho=1025, g=9.8, mf=4866, mb=2433, ma=1335.535, k=80000, l=0.5, ht=0.8, f=6250, lamda=151.4388):
    y1, y2, z1, z2 = y

    dy1 = y2
    dz1 = z2

    dy2 = (f*cos(omega*t)-(mf+ma)*g+k*(z1-y1-ht-l)-K*(z2-y2)-lamda*y2)/(ma+mf)
    # dy2=(f*cos(omega*t)-(mf+ma)*g+k*(z1-y1-ht-l))/(ma+mf)

    dz2 = (-mb*g-k*(z1-y1-ht-l)+K*(z2-y2))/mb
    # dz2=(-mb*g-k*(z1-y1-ht-l))/mb

    return [dy1, dy2, dz1, dz2]

y0 = [-1.298, 0, -0.191, 0]
t = np.linspace(0, 10, 100)
sol1 = odeint(Model1, y0, t)
sol1 = np.log(sol1)

plt.plot(t, sol1[:, 1], 'b', label='y2(t)')
plt.plot(t, sol1[:, 3], 'g', label='z2(t)')

plt.legend(loc='best')
plt.xlabel('t')
plt.grid()

plt.show()

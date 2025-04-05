#!/usr/bin/env python3

import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.time import Time
from scipy.optimize import fsolve

#Variables globales

# Parámetros dados
a = 1.30262 * R_earth  # Semieje mayor en m
a = a.to('km') # Semieje mayor en km
e = 0.16561  # Excentricidad
w = 15 * u.deg  # Argumento del pericentro
t_p = "2025-03-31 00:00:00"  # Tiempo en el que pasa por el pericentro (E=0)

# Constante gravitacional terrestre
GM_earth = 398600.4405 * (u.km**3 / u.s**2)

#Se convierte el tiempo a segundos
t_p = Time(t_p, format='iso', scale='utc')




# Ecuación de Kepler:
def keplerEquation(E, l):
    return E - e * np.sin(E) - l

def bisectionMethod(a, b, l): #a: inferiorLimit ,  b:superiorLimit
    #Falta justificar por qué siempre funciona con los a y b elegidos.
    maxError = np.sqrt(1e-16)
    fa = keplerEquation(a,l)
    while (b-a)>maxError:
        m = (a+b)/2.0
        fm = keplerEquation(m,l)

        if (fm*fa<0):
            b=m
        else:
            a=m
            fa=fm
    return (a+b)/2

def position(t):
    t = Time(t, format='iso', scale='utc')
    delta_t = (t-t_p).to(u.s)
    l = np.sqrt(GM_earth / a**3) * delta_t
    lValue = l.value
    leftLimit = lValue - lValue/2
    rightLimit = lValue + lValue/2
    E = bisectionMethod(leftLimit,rightLimit,l) #Anomalía excéntrica

    # E0 = lValue
    # E_solution = fsolve(keplerEquation, E0, args=(lValue,))

    # print(E)
    # print(E_solution)

    f = 2*np.arctan(np.tan(E/2)*np.sqrt((1+e)/(1-e)))
    r = a*(1-np.power(e,2))/(1+e*np.cos(f))
    f=f*u.rad
    f=f.to(u.deg)
    phi = f + w
    return r,phi


def main():
    t_test = "2025-04-01 00:00:00"  # Tiempo de prueba.
    t_test = Time(t_test, format='iso', scale='utc')
    t = t_test
    r,phi = position(t)
    print(r,phi)

    # epsilon = 1.

    # while (1.+epsilon != 1.):
    #     epsilon = epsilon/2.

    # print(epsilon)



if __name__ == "__main__":
    main()


#!/usr/bin/env python3

import numpy as np
from astropy import units as u
from astropy.constants import R_earth, GM_earth
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve

#Variables globales

# Parámetros dados
a = 1.30262 * R_earth  # Semieje mayor en m
a = a.to('km') # Semieje mayor en km
e = 0.16561   #Excentricidad
w = 15* u.deg  # Argumento del pericentro
w = w.to(u.rad)
t_p = "2025-03-31 00:00:00"  # Tiempo en el que pasa por el pericentro (E=0)
t_p = Time(t_p, format='iso', scale='utc') #Se convierte el tiempo inicial a formato Time
GM_earth = GM_earth.to(u.km**3 / u.s**2) # Constante gravitacional terrestre con masa terrestre
T_o = 2*np.pi*np.sqrt(np.power(a,3)/GM_earth) #Periodo


# Ecuación de Kepler:
def keplerEquation(E, l):
    return E - e * np.sin(E) - l

def bisectionMethod(l):
    #Falta justificar por qué siempre funciona con los a y b elegidos.
    l = l.value
    a = l - l/2
    b = l + l/4

    maxError = np.sqrt(1e-16)
    fa = keplerEquation(a,l)
    fb = keplerEquation(b,l)
    while (b-a)>maxError:

        m = (a+b)/2.0
        fm = keplerEquation(m,l)
        if (fm*fa<0):
            b=m
        else:
            a=m
            fa=fm
        fb = keplerEquation(b,l)

    return (a+b)/2

def vectorizedBisectionMethod(l):
    #Falta justificar por qué siempre funciona con los a y b elegidos.

    l = l.value
    a = l - l/2
    b = l + l/4

    maxError = np.sqrt(1e-16)
    fa = keplerEquation(a,l)
    while not np.all((b - a) < maxError):
        m = (a + b) / 2.0
        fm = keplerEquation(m, l)

        mask = fm * fa < 0  # En qué lugares actualizar b, para evaluar vectores
        b = np.where(mask, m, b)
        a = np.where(mask, a, m)
        fa = np.where(mask, fa, fm)

    return (a + b) / 2.0  # Devuelve array de soluciones E

def arctanMod(x):
    invTan = np.arctan(x)
    invTan = np.where(invTan>=0,invTan, np.pi+invTan)
    return invTan

def position(t):
    delta_t = (t-t_p).to(u.s)
    l = np.sqrt(GM_earth / a**3) * delta_t #Arreglo de l
    E = vectorizedBisectionMethod(l) #Anomalía excéntrica
    # f = 2*np.arctan(np.tan(E/2)*np.sqrt((1+e)/(1-e)))
    f = 2*arctanMod(np.tan(E/2)*np.sqrt((1+e)/(1-e))) #Se modifica la arcotangente para garantizar la continuidad. (Soluciona la discontinuidad dada por la tangente)
    r = a*(1-np.power(e,2))/(1+e*np.cos(f))
    f=f*u.rad
    phi = f + w
    return r,phi,f

def orbit():

    T = TimeDelta((T_o).value*u.s)


    T = t_p + T


    # Diferencia total de tiempo como TimeDelta
    delta_total = T - t_p

    steps = np.linspace(0, 1, 53)
    delta_array = steps * delta_total  # TimeDelta array

    # Sumar a la fecha inicial para obtener el array de fechas
    t = t_p + delta_array

    r,phi,f = position(t)


    plt.style.use('seaborn-v0_8')


    # x e y en coordenadas cartesianas, en sistema rotado.
    xp = r * np.cos(f)
    yp = r * np.sin(f)

    #x e y en coordenadas cartesianas en sistema no rotado. Es equivalente a la transformación x= xp*np.cos(w) - yp*np.sin(w) e y= xp*np.sin(w) + yp*np.cos(w)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    t_vals = (t - t[0]).to_value('s')  # Tiempo desde t0 en segundos


    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=False)
    # 1. Órbita (x vs y)
    axs[0].plot(x, y, label='y vs x', color='teal', linewidth=2, marker='o')
    axs[0].set_title('Órbita del satélite a lo largo de un periodo.')
    axs[0].set_xlabel('x [km]')
    axs[0].set_ylabel('y [km]')
    axs[0].legend()

    # Marcar el punto (0,0) como ubicación Tierra - foco
    axs[0].scatter(0, 0, color='blue', marker='o', s=80)
    axs[0].annotate('Tierra - foco', (0, 0), textcoords="offset points", xytext=(10, -15), ha='left', fontsize=9)

    # Marcar el pericentro (x[0], y[0])

    axs[0].scatter(x[0].value, y[0].value, color='red', marker='*', s=100)
    axs[0].annotate('Pericentro', (x[0].value, y[0].value), textcoords="offset points", xytext=(5, 10), ha='left', fontsize=9)
    axs[0].set_aspect('equal', adjustable='datalim')

    axs[0].ticklabel_format(style='scientific', axis='both', scilimits=(-1,3))

    # Leyenda
    axs[0].legend()



    # 2. r vs t
    axs[1].plot(t_vals, r, label='r vs t', color='darkorange', linewidth=2, marker='.')
    axs[1].set_title('r vs t')
    axs[1].set_xlabel('Tiempo [s]')
    axs[1].set_ylabel('r [km]')
    axs[1].ticklabel_format(style='scientific', axis='both', scilimits=(-1,3))
    axs[1].legend()

    # 3. phi vs t
    axs[2].plot(t_vals[:-1], phi[:-1], label=r'$\phi$ vs t', color='purple', linewidth=2, marker='o')
    axs[2].set_title(r'$\phi$ vs t')
    axs[2].set_xlabel('Tiempo [s]')
    axs[2].set_ylabel(r'$\phi$ [rad]')
    axs[2].ticklabel_format(style='scientific', axis='both', scilimits=(-1,3))
    axs[2].legend()

    # Ajustar diseño y guardar
    plt.tight_layout()
    plt.savefig('orbita_completa.pdf')

def date(r0):
    # posición radial y retorna tiempo t0 en que el satélite se localiza allí
    pass




def main():
    t_test = "2025-04-01 00:00:00"  # Tiempo de prueba.
    t_test = "2025-03-31 01:02:48.293"
    t_test = Time(t_test, format='iso', scale='utc')
    t = t_test





    # r,phi,f = position(t)
    # print(r, phi.to(u.deg))
    orbit()






if __name__ == "__main__":
    main()


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

def rMaxMin():
    fmax = 2*arctanMod(np.tan(np.pi/2)*np.sqrt((1+e)/(1-e))) #Se modifica la arcotangente para garantizar la continuidad. (Soluciona la discontinuidad dada por la tangente)
    fmin = 2*arctanMod(np.tan(0/2)*np.sqrt((1+e)/(1-e))) #Se modifica la arcotangente para garantizar la continuidad. (Soluciona la discontinuidad dada por la tangente)
    rmin = round((a*(1-np.power(e,2))/(1+e*np.cos(fmin))).value,3)
    rmax = round((a*(1-np.power(e,2))/(1+e*np.cos(fmax))).value,3)
    return rmax, rmin

def findIndexDate(r,r0):
    halfIndex= int(len(r)/2)
    rC = np.array(r[:halfIndex]) #rCreciente
    rD = np.array(r[halfIndex:]) #rDecreciente

    # t_vals = (t - t[0]).to_value('s')  # Tiempo desde t0 en segundos
    # plt.plot(t_vals[:halfIndex], rC, label='r vs t', linewidth=2, marker='x')
    # plt.plot(t_vals[halfIndex:], rD, label='r vs t', linewidth=2, marker='*')
    # plt.plot(t_vals, r, label='r vs t', color='darkorange', linewidth=2, marker='.')
    # plt.savefig('control.pdf')

    if not np.all(np.diff(rC) >= 0): #Arreglo creciente.
        raise ValueError("El arreglo r no está en orden creciente")
    if not np.all(np.diff(rD) <= 0): #Arreglo decreciente.
        raise ValueError("El arreglo r no está en orden decreciente")


    idx1 = np.searchsorted(rC, r0)  #Se busca el índice a derecha, tal que r0 quede ordenado
    idx2 = np.searchsorted(-rD, -r0)  #Se busca el índice a derecha, tal que r0 quede ordenado

    #Se asegura que se hayan encontrado los índices
    if idx1 == 0 or idx1 == len(rC):
        raise ValueError("r0 está fuera del rango del arreglo rC")

    if idx2 == 0 or idx2 == len(rD):
        raise ValueError("r0 está fuera del rango del arreglo rD")

    # print(r[idx1-1],r0,r[idx1], idx1-1)
    # print(r[len(rC)+idx2-1],r0,r[len(rC)+idx2], len(rC)+idx2-1)

    return idx1-1, len(rC)+idx2-1

def quadraticInterpolation(x1, x2, x3, f1, f2, f3, x):
    p2 = (((x-x2)*(x-x3))/((x1-x2)*(x1-x3)))*f1 + (((x-x1)*(x-x3))/((x2-x1)*(x2-x3)))*f2 + (((x-x1)*(x-x2))/((x3-x1)*(x3-x2)))*f3
    return p2

def date(r0):
    # posición radial y retorna tiempo t0 en que el satélite se localiza allí
    T = TimeDelta((T_o).value*u.s)
    T = t_p + T
    # Diferencia total de tiempo como TimeDelta
    delta_total = T - t_p
    steps = np.linspace(0, 1, 100)
    delta_array = steps * delta_total  # TimeDelta array

    # Sumar a la fecha inicial para obtener el array de fechas
    t = t_p + delta_array

    r,phi,f = position(t)

    rmax, rmin = rMaxMin()

    # rmax= round(np.max(r).value,3)
    # rmin= round(np.min(r).value,3)

    if r0>rmax or r0<rmin:
         raise ValueError(f"El valor de r0 debe estar entre {rmin} km y {rmax} km")

    rIndexIncreasing, rIndexDecreasing = findIndexDate(r,r0)

    t_vals = (t - t[0]).to_value('s')  # Tiempo desde t0 en segundos


    # def findDateGivenIndex(t_vals,rIndex):

    x1,x2,x3 = t_vals[rIndexIncreasing], t_vals[rIndexIncreasing+1], t_vals[rIndexIncreasing+2]
    f1,f2,f3 = r[rIndexIncreasing], r[rIndexIncreasing+1], r[rIndexIncreasing+2]

    x = np.linspace(x1,x2,30)
    r_interpolated = np.array(quadraticInterpolation(x1, x2, x3, f1, f2, f3,x))





    closerIndex = np.argmin(np.abs(r_interpolated - r0)) #índice para el cual más nos acercamos
    print(r_interpolated[closerIndex])
    print(r0)

    closerT = x[closerIndex] #Tiempo para el cual está en r0, medido desde tp=0

    print("closerT: ",closerT)
    print(x1)


    date = t_p + closerT*u.s
    print(t_p.iso)
    print((t_p+x1*u.s).iso)
    print(date.iso)

    # Crear el gráfico
    plt.figure(figsize=(8, 5))

    # Curva interpolada
    plt.plot(x, r_interpolated,label='Interpolación cuadrática', color='teal')

    # Puntos originales
    plt.scatter([x1, x2, x3], [f1.value, f2.value, f3.value], color='red', label='Datos originales', zorder=5)

#    plt.plot(t_vals[(rIndexIncreasing-1):(rIndexIncreasing+3)], r[(rIndexIncreasing-1):(rIndexIncreasing+3)], label='r vs t', color='darkorange', marker='.')
    plt.plot(t_vals, r, label='r vs t', color='darkorange', marker='.')

    # Línea vertical en x = closerT
    plt.axvline(x=closerT, color='gray', linestyle='--', label='x = closerT')

    # Línea horizontal en y = r0
    plt.axhline(y=r0, color='purple', linestyle='--', label='r = r0')

    # Etiquetas, leyenda y guardado
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Distancia [km]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'controlInterpolation.pdf')


    # print(t.iso)













def main():
    t_test = "2025-04-01 00:00:00"  # Tiempo de prueba.
    t_test = Time(t_test, format='iso', scale='utc')
    t = t_test





    # r,phi,f = position(t)
    # print(r, phi.to(u.deg))
#    orbit()
    date(9000)





if __name__ == "__main__":
    main()


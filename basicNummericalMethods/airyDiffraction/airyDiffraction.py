#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # Función de Bessel de SciPy para comprobar.


def bessel_m(m, x, num_points=1001):

    # num_points debe ser impar para siempre encontrar tríos de puntos.
    if num_points % 2 == 0:
        num_points += 1

    a, b = 0, np.pi #Límites de integración

    theta = np.linspace(a, b, num_points) #Puntos de integración
    h = (b - a) / (num_points - 1) #Tamaño del paso (Cantidad de intervalos siempre es num_points-1)

    f_values = np.cos(m * theta - x * np.sin(theta))

    #Regla de Simpson (h/3)*[f(x_0)+2*suma f(x_pares sin tomar el último)+4* suma f(x_impares)+ f(x_n)]
    #Forma de f_values = [f0,f1,f2,f3,...,f997,f998,f999] len(f_values)=num_points
    integral =(h/3)*(f_values[0]
                    + 2 * np.sum(f_values[2:-1:2])
                    + 4 * np.sum(f_values[1::2]) + f_values[-1])
    integral = integral / np.pi

    return integral


def vectorized_bessel_m(m, x, num_points=1001):

    # num_points debe ser impar para siempre encontrar tríos de puntos.
    if num_points % 2 == 0:
        num_points += 1

    a, b = 0, np.pi #Límites de integración

    theta = np.linspace(a, b, num_points) #Puntos de integración
    h = (b - a) / (num_points - 1) #Tamaño del paso (Cantidad de intervalos siempre es num_points-1)

    x_array = np.atleast_1d(x) #Asegura soporte de escalares y arreglos (definición: Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.)


    '''
    np.outer devuelve matriz 2D con M_{i,j} = x_array[i]*sin(theta[j]): Al cambiar de filas, varío el valor de x y al cambiar columnas, varío el valor de theta.
    Al restarle m*theta a la matriz 2D, se le resta a cada fila (que tiene x fijo) el valor m*theta.
    Se obtiene todas las combinaciones de x_array y theta
    '''
    integrand = np.cos(m * theta - np.outer(x_array, np.sin(theta)))


    '''
    La antigua regla de Simpson implementada en bessel_m():
    #Regla de Simpson (h/3)*[f(x_0)+2*suma f(x_pares sin tomar el último)+4* suma f(x_impares)+ f(x_n)]
    #Forma de f_values = [f0,f1,f2,f3,...,f997,f998,f999] len(f_values)=num_points

    integral =(h/3)*(f_values[0]
    + 2 * np.sum(f_values[2:-1:2]) #Suma de índices pares.
    + 4 * np.sum(f_values[1::2]) #Suma de índices impares.
    + f_values[-1])

    '''



    integral = h/3* (integrand[:, 0] #Equivalente a tomar x=0 de la matriz 2D: f(x_0) para todos los theta.
        + 2 * np.sum(integrand[:, 2:-1:2], axis=1)  # Suma de los índices pares. axis=1 suma a lo largo de las columnas (con x fijo ya que varía es con las filas)
        + 4 * np.sum(integrand[:, 1::2], axis=1)  # Suma de los índices impares. Similar a la suma de pares.
                     + integrand[:, -1] #Equivalente a tomar x=n de la matriz 2D: f(x_n) para todos los theta.
                    )

    integral = integral / np.pi

    # Si x_array es un escalar, se devuelve el escalar. En caso contrario, se devuelve el arreglo para los distintos x.
    return integral[0] if x_array.shape[0] == 1 else integral




def graphicBessel():
    x=np.linspace(0,20,1000)
    m_array=[0,1,2]


    plt.style.use('seaborn-v0_8')
    # Graficar
    plt.figure(figsize=(8, 5))
    for m in m_array:
        y0 = vectorized_bessel_m(m,x)
        plt.plot(x, y0, label=f"$J_{m}(x)$")
        # plt.plot(x, jv(m, x), label=f"Scipy: $J_{m}(x)$",linestyle="--", color="black")


    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.title(f"Funciones de Bessel de primer tipo $J_{m_array[0]}(x),J_{m_array[1]}(x)$ y $J_{m_array[2]}(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("besselOrden1_2_3.pdf")

def main():

    # m = 1  #Orden Bessel
    # x = 5  #Valor de prueba
    # numPoints = 1001 #Debe ser un número impar
    # result = vectorized_bessel_m(m, x,numPoints)
    # result2 = jv(m,x)
    # print(f"J_{m}({x}) ≈ {result}")
    # print(f"Scipy: J_{m}({x}) ≈ {result2}")
    graphicBessel()



if __name__ == "__main__":
    main()

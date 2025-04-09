#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    print("besselOrden1_2_3.pdf creado.")


def findMaxMin(I,r):

    dr = np.diff(r) #Se calcula r[i+1]-r[i]
    dr = np.mean(dr)  #Se usa esto como el paso promedio, aunque en este caso están uniformemente espaciados.

    # Derivada central: (f(x + dx) - f(x - dx))/(2*dx)
    # I[1,2,3,4] -> I[2:]=[3,4] e I[:-2]=[1,2] -> I[2:]-I[:-2] = [3-1,4-2] = f(x + dx) - f(x - dx) para todo punto menos extremos.

    dI_dr = (I[2:]-I[:-2])/(2*dr)


    '''
    np.sign devuelve un arreglo donde vale 1 si el valor es positivo, -1 si el valor es negativo y 0 para los ceros.
    Si hay un cambio de signo se tendrá -1-1=-2 (máximo) ó 1+1=2 (mínimo), que son distintos de 0. En caso de que no haya cambio de signo, se tendrá 0.

    '''

    index_change = np.diff(np.sign(dI_dr))

    maxima_index = np.where(index_change == -2)[0]+1  #Se suma 1 por el corrimiento de dI_dr respecto a I

    minima_index = np.where(index_change == 2)[0]+1  #Se suma 1 por el corrimiento de dI_dr respecto a I


    r_minima = r[minima_index]
    r_maxima = r[maxima_index]
    I_minima = I[minima_index]
    I_maxima = I[maxima_index]

    #Se redefinen r e I por cuestiones de visualización.
    r=r[minima_index[0]:]
    I=I[minima_index[0]:]

    plt.style.use('seaborn-v0_8')
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(r, I, label="I vs r")
    ax1.plot(r_maxima, I_maxima, "ro", label=r"$r_{max}$",markersize=5)
    ax1.plot(r_minima, I_minima, "bo", label=r"$r_{min}$",markersize=5)
    ax1.set_xlabel(r"r [$\mu m$]")
    ax1.set_ylabel("Intensidad [u.a.]")
    ax1.set_title("Máximos y mínimos de I vs r")
    ax1.ticklabel_format(style='scientific', axis='both', scilimits=(-3,3))
    ax1.legend()
    ax1.grid(True)
    fig1.savefig('IMaxMin.pdf')
    print('\n IMaxMin.pdf creado. \n')
    print(f"Los valores de r donde se ubican los máximos de intensidad son r={r_maxima} micrómetros. \n")
    return maxima_index, minima_index

def graphicI(waveLambda,rMaxValue):
    r = np.linspace(1e-3,rMaxValue,200)
    k = 2*np.pi/waveLambda
    x = k*r
    J1 = vectorized_bessel_m(1,x)
    I = np.power(2*J1/x,2)

    maxima_index, minima_index = findMaxMin(I,r)

    r_minima = r[minima_index]
    r_maxima = r[maxima_index]

    # Configuración del gráfico polar
    theta = np.linspace(0, 2 * np.pi, len(r))
    r_grid, theta_grid = np.meshgrid(r, theta)


    '''
    Se normaliza a escala logarítmica para que pequeños cambios en I, se vean representados
    en grandes cambios para I_normalized.
    Se evidenció que solo con I, se toma gran parte de los valores de intensidad como 0 y no
    se evidencia el patrón de difracción.
    '''
    I_normalized = np.log10(I + 1e-10)  # Se suma un término pequeño para evitar errores con log(0)
    I_normalized -= np.min(I_normalized)  # Mínimo en 0
    I_normalized /= np.max(I_normalized)  # Máximo en 1

    I_polar_normalized = np.tile(I_normalized, (len(theta), 1)) # Repetir la intensidad para todos los ángulos.

    plt.style.use('seaborn-v0_8') #Gráfico polar con la intensidad normalizada

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))

    c = ax.contourf(theta_grid, r_grid, I_polar_normalized,levels=100, cmap='viridis')
    # Mínimos y máximos locales
    ax.plot(np.zeros_like(r_minima), r_minima, 'bo', label='\n'+r'$r_{min}$',markersize=5)
    ax.plot(np.zeros_like(r_maxima), r_maxima, 'ro', label=r'$r_{max}$',markersize=5)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0, 0.9))

    #Etiquetas de radio.
    ax.set_rticks([])
    for radius in np.linspace(r.min(), r.max(), 5):
        ax.text(45, radius, f"{radius:.2f}", ha='center', va='center',
            color='black', fontsize=10, bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))

    cbar = plt.colorbar(c, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Intensidad (Normalizada Log)')
    plt.title(r'Patrón de difracción circular: I($r$ [$\mu m$])', va='bottom')
    fig.savefig('patronDifraccion.pdf')
    print('patronDifraccion.pdf creado.')



def main():

    # m = 1  #Orden Bessel
    # x = 5  #Valor de prueba
    # numPoints = 1001 #Debe ser un número impar
    # result = vectorized_bessel_m(m, x,numPoints)
    # print(f"J_{m}({x}) ≈ {result}")

    graphicBessel()
    waveLambda = 0.5 # microm = 500 nm
    rMaxValue = 2 # microm
    graphicI(waveLambda,rMaxValue) #Se calcula el valor de los máximos en findMaxMin



if __name__ == "__main__":
    main()

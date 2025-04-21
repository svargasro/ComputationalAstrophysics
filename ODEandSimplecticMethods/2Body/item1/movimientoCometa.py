#!/usr/bin/env python3
import numpy as np
from astropy.constants import M_sun,G
from astropy.units import au, yr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

#Constantes globales
solar_mass = M_sun # Masa solar (Nueva unidad de masa)
a_unit = au # (Nueva unidad de distancia)
a_unit = a_unit.to('m')
yearToSeg = yr.to('s')  # (juliano)

#Con 1newM = 1 u.a. ; 1newKg = 1 masa solar ; 1news = 1 año -> La constante gravitacional queda dada por:
G_sun = G*solar_mass*np.power(yearToSeg,2)*(1/np.power(a_unit,3)) #~ 4*pi^2
G_sun = G_sun.value

num_orbitas = 5 #Número de órbitas a graficar.

def dvdt(x, y):
        r = np.sqrt(x**2 + y**2)
        shared = -G_sun/(r**3)
        ax = shared*x
        ay = shared*y
        return ax, ay


def RK4(x0,y0,vx0,vy0,tf,dt):

    n = int(tf/dt)

    #Inicialización de arreglos:
    t = np.linspace(0, tf, n + 1)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    vx = np.zeros(n + 1)
    vy = np.zeros(n + 1)
    energy = np.zeros(n + 1)
    angularMom = np.zeros(n + 1)
    # Condiciones iniciales
    x[0], y[0] = x0, y0
    vx[0], vy[0] = vx0, vy0

    # Método de Runge-Kutta de cuarto orden, energía y momento angular.
    for i in range(n):
        xi, yi, vxi, vyi = x[i], y[i], vx[i], vy[i]
        speed2 = np.power(vxi,2)+np.power(vyi,2)
        r = np.sqrt(np.power(xi,2)+np.power(yi,2))
        energy[i] =  speed2/2 - G_sun/r
        angularMom[i] = xi*vyi - vxi*yi

        # k1
        k1x = vxi * dt #dxdt=vxi. No se usa una función adicional porque devolvería vxi.
        k1y = vyi * dt
        #No se usa dvdt() para aprovechar el cálculo de r.
        sharedk1xk1y = -G_sun/(r**3)
        ax = sharedk1xk1y*xi
        ay = sharedk1xk1y*yi
        k1vx = ax * dt
        k1vy = ay * dt

        # k2
        k2x = (vxi + 0.5 * k1vx) * dt
        k2y = (vyi + 0.5 * k1vy) * dt
        ax, ay = dvdt(xi + 0.5 * k1x, yi + 0.5 * k1y)
        k2vx = ax * dt
        k2vy = ay * dt

        # k3
        k3x = (vxi + 0.5 * k2vx) * dt
        k3y = (vyi + 0.5 * k2vy) * dt
        ax, ay = dvdt(xi + 0.5 * k2x, yi + 0.5 * k2y)
        k3vx = ax * dt
        k3vy = ay * dt

        # k4
        k4x = (vxi + k3vx) * dt
        k4y = (vyi + k3vy) * dt
        ax, ay = dvdt(xi + k3x, yi + k3y)
        k4vx = ax * dt
        k4vy = ay * dt

        # Actualización de valores.
        x[i + 1] = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y[i + 1] = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        vx[i + 1] = vxi + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6
        vy[i + 1] = vyi + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6

        if (i==(n-1)): #If necesario para incluir el valor final de la energía.
                xi, yi, vxi, vyi = x[i+1], y[i+1], vx[i+1], vy[i+1]
                speed2 = np.power(vxi,2)+np.power(vyi,2)
                r = np.sqrt(np.power(xi,2)+np.power(yi,2))
                energy[i+1] = speed2/2 - G_sun/r
                angularMom[i+1] = xi*vyi - vxi*yi
    print("iteraciones no adaptativo: ", n)
    return t,x,y,energy,angularMom

def RK4step(xi,yi,vxi,vyi,dt): #Se define solo un paso del Runge-Kutta

    r = np.sqrt(np.power(xi,2)+np.power(yi,2))
    # k1
    k1x = vxi * dt #dxdt = vxi
    k1y = vyi * dt #dydt = vyi

    #No se usa dvdt() para aprovechar el cálculo de r.
    sharedk1xk1y = -G_sun/(r**3)
    ax = sharedk1xk1y*xi
    ay = sharedk1xk1y*yi
    k1vx = ax * dt
    k1vy = ay * dt

    # k2
    k2x = (vxi + 0.5 * k1vx) * dt
    k2y = (vyi + 0.5 * k1vy) * dt
    ax, ay = dvdt(xi + 0.5 * k1x, yi + 0.5 * k1y)
    k2vx = ax * dt
    k2vy = ay * dt

    # k3
    k3x = (vxi + 0.5 * k2vx) * dt
    k3y = (vyi + 0.5 * k2vy) * dt
    ax, ay = dvdt(xi + 0.5 * k2x, yi + 0.5 * k2y)
    k3vx = ax * dt
    k3vy = ay * dt

    # k4
    k4x = (vxi + k3vx) * dt
    k4y = (vyi + k3vy) * dt
    ax, ay = dvdt(xi + k3x, yi + k3y)
    k4vx = ax * dt
    k4vy = ay * dt

    xf = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    yf = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    vxf = vxi + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6
    vyf = vyi + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6

    return xf, yf, vxf, vyf


def RK4AdaptativeStep(x0,y0,vx0,vy0,tf,dt):

    n = int(tf/dt)


    #Inicialización de arreglos. A lo sumo tendrán esa cantidad de elementos, para este problema.
    t_array = np.zeros(n + 1)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    vx = np.zeros(n + 1)
    vy = np.zeros(n + 1)
    energy = np.zeros(n + 1)
    angularMom = np.zeros(n + 1)

    # Condiciones iniciales
    x[0], y[0] = x0, y0
    vx[0], vy[0] = vx0, vy0

    r0 = np.sqrt(x0**2 + y0**2)
    speed2 = np.power(vx0,2)+np.power(vy0,2)
    energy[0] =  speed2/2 - G_sun/r0
    angularMom[0] = x0*vy0 - vx0*y0

    #Valor de la precisión.
    epsilon = 1e-10


    # Método de Runge-Kutta de cuarto orden, energía y momento angular.
    t=0
    i= 0 #índice para ingresar información en los arreglos.
    iterCount = 0 #Contador de iteraciones.

    while(t <= tf):
        # print(i)
        #Se calcula x1,y1,vx1,vy1 en dos pasos de dt.
        x1_mid,y1_mid,vx1_mid,vy1_mid = RK4step(x0,y0,vx0,vy0,dt)
        x1,y1,vx1,vy1 = RK4step(x1_mid,y1_mid,vx1_mid,vy1_mid,dt)

        #Se calcula x2,y2,vx2,vy2 en un paso de 2dt
        x2,y2,vx2,vy2 = RK4step(x0,y0,vx0,vy0,2*dt)

        #Se usa el valor de la distancia como indicador del comportamiento de x e y.
        r1 = np.sqrt(x1**2 + y1**2)
        r2 = np.sqrt(x2**2 + y2**2)


        theta = np.abs(r2-r1)*(1/(30*epsilon*dt))

        if theta<=1:
                #Se toma la solución aceptada de x1,y1,vx1,vy1 por tener pasos más pequeños y por tanto mejor precisión.
                x[i + 1] = x1
                y[i + 1] = y1
                vx[i + 1] = vx1
                vy[i + 1] = vy1
                t += 2*dt
                t_array[i + 1] = t
                x0,y0,vx0,vy0 = x1,y1,vx1,vy1 #Actualización de valores para siguiente iteración.
                speed2 = np.power(vx1,2)+np.power(vy1,2)
                energy[i+1] =  speed2/2 - G_sun/r1
                angularMom[i+1] = x1*vy1 - vx1*y1
                i += 1
        if theta != 0:
                dt = dt*np.power(theta,-1/4)
        else:
                dt = 1e-2 #Valor que se toma dt en caso de que no haya diferencia entre iteraciones.

        iterCount += 1
    print("iteraciones adaptativo: ",iterCount)

    limitIndex = int(i+1)

    t_array = t_array[:limitIndex]
    x = x[:limitIndex]
    y = y[:limitIndex]
    energy = energy[:limitIndex]
    angularMom = angularMom[:limitIndex]
    return t_array,x,y,energy,angularMom


def graphicsRK4(x,y,t,energy,angularMom,isAdaptative):

    # Crear la figura con 3 subgráficos
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # Plot 1: y vs x
    axs[0].plot(x, y, label="Trayectoria")
    axs[0].set_title("Trayectoria: y vs x")
    axs[0].set_xlabel("x [UA]")
    axs[0].set_ylabel("y [UA]")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: energy vs t
    axs[1].plot(t, energy, label="Energía", color='b')
    axs[1].set_title("Energía vs tiempo")
    axs[1].set_xlabel("Tiempo (t) [yr]")
    axs[1].set_ylabel(r'Energía [$UA^2/yr^2$]')
    axs[1].set_ylim(np.min(energy)-0.05,np.max(energy)+0.05)
    axs[1].grid(True)
    axs[1].legend()
    #print("E: ",np.min(energy),np.max(energy),np.mean(energy),np.std(energy))


    # Plot 3: angularMom vs t
    axs[2].plot(t, angularMom, label="Momento angular", color='r')
    axs[2].set_title("Momento angular vs tiempo")
    axs[2].set_xlabel("Tiempo (t) [yr]")
    axs[2].set_ylabel("Momento angular []")
    axs[2].set_ylim(np.min(angularMom)-0.1,np.max(angularMom)+0.1)
    axs[2].grid(True)
    axs[2].legend()
    #print("MA: ",np.min(angularMom),np.max(angularMom),np.mean(angularMom),np.std(angularMom))


    plt.tight_layout()
    if isAdaptative:
        file_name = "orbitaRK4_adaptativo.pdf"
    else:
        file_name = "orbitaRK4.pdf"

    plt.savefig(file_name)
    print(file_name+" creado.")
    plt.close()

    num_points = len(t)

    if isAdaptative:
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        # Plot x vs t
        axs[0].plot(t, x, linestyle='-', color='blue', label='x vs t')
        axs[0].set_title(r"x vs t. Comportamiento paso $\Delta t$")
        for i in range(num_points):
            axs[0].axvline(x=t[i], color='gray', linestyle='-', alpha=0.1)
        axs[0].set_xlabel('t [yr]')
        axs[0].set_ylabel('x [UA]')
        axs[0].legend(loc='upper right')

        # Plot y vs t
        axs[1].plot(t, y, linestyle='-', color='red', label='y vs t')
        axs[1].set_title(r"y vs t. Comportamiento paso $\Delta t$")
        for i in range(num_points):
            axs[1].axvline(x=t[i], color='gray', linestyle='-', alpha=0.1)
        axs[1].set_xlabel('t [yr]')
        axs[1].set_ylabel('y [UA]')
        axs[1].legend(loc='upper right')

        # Guardar el gráfico
        plt.savefig("xyRK4_adaptativo.pdf")
        print("xyRK4_adaptativo.pdf creado.")
        plt.close()




def main():

    #Como dos objetos orbitando definen un único plano, podemos trabajar el problema en dos dimensiones.
    #Las unidades utilizadas se describen como constantes globales al inicio del código.
    x0 = (4e12)/a_unit
    vy0 = (500/a_unit)*yearToSeg
    y0,vx0 = 0,0
    finalTime = 49.5*num_orbitas #Periodo es aproximadamente 49.5 años
    dt = 0.0002 #0.0002


    t,x,y,energy,angularMom = RK4(x0,y0,vx0,vy0,finalTime,dt) #Runge-Kutta usual

    t_adap,x_adap,y_adap,energy_adap,angularMom_adap = RK4AdaptativeStep(x0,y0,vx0,vy0,finalTime,dt) #Runge-Kutta de paso adaptativo


    graphicsRK4(x,y,t,energy,angularMom,False)
    graphicsRK4(x_adap,y_adap,t_adap,energy_adap,angularMom_adap,True)

    '''
    ¿Como se comparan los resultados obtenidos con los dos métodos?

    El método de Runge-Kutta con paso adaptativo es muchísimo más eficiente en términos
    de tiempo de cómputo en comparación con el método de Runge-Kutta 4 estándar.
    Además, para el método de paso adapativo no es necesario un dt inicial tan pequeño como para
    el método estándar para obtener las órbitas elípticas, puesto que este se ajusta inicialmente.

    En términos de precisión, el método adaptativo puede mostrar limitaciones cuando
    el cuerpo celeste se encuentra muy alejado del Sol, ya que las pequeñas
    variaciones en fuerzas gravitatorias pueden no capturarse. Sin embargo,
    esta precisión es más que suficiente para analizar el comportamiento general
    del sistema en la mayoría de los casos prácticos.

    Si se requiere mayor precisión en escenarios específicos,
    como en los puntos más alejados de la órbita,
    es posible ajustar el valor de epsilon (tolerancia del error local)
    en el método adaptativo para alcanzar el nivel deseado de exactitud.
    '''




if __name__ == "__main__":
    main()

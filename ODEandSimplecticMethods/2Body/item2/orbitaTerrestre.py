#!/usr/bin/env python3
import numpy as np
from astropy.constants import M_sun,G, M_earth
from astropy.units import au, yr, hour
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Calcula los segundos en una hora

#Constantes globales
N=2
solar_mass = M_sun # Masa solar (Nueva unidad de masa)
a_unit = au # (Nueva unidad de distancia)
a_unit = a_unit.to('m')
yrToSec = yr.to('second') #Hora en segundos

#Con 1newM = 1 u.a. ; 1newKg = 1 masa solar ; 1news = 1 hora -> La constante gravitacional queda dada por:
G_sun = G*solar_mass*np.power(yrToSec,2)*(1/np.power(a_unit,3))
G_sun = G_sun.value


class Cuerpo:
    def __init__(self,x0,y0,z0,Vx0,Vy0,Vz0,m0):
        self.m=m0
        self.r=np.array([x0,y0,z0])
        self.V=np.array([Vx0,Vy0,Vz0])

    def resetForce(self):
        self.F = np.array((0,0,0))

    def addForce(self,dF):
        self.F = self.F + dF


class dynamicManager:
    def __init__(self):
        self.EpTotal = 0.0

    def calculateForceBetween(self,Planeta1, Planeta2):

        m1=Planeta1.m
        m2=Planeta2.m
        r21 = Planeta2.r-Planeta1.r #Vector dirigido del cuerpo 1 al cuerpo 2. #(x,y,z)
        r_mag= np.linalg.norm(r21) #sqrt(x**2 + y**2+ z**2)
        aux = G_sun*m2*m1*pow(r_mag,-3)
        F1 = r21*aux
        Planeta1.addForce(F1) #Siente fuerza hacia el cuerpo 2.
        #No se implementa la fuerza que siente Planeta2, que corresponde al sol, puesto que no se pide en el problema.
        #Planeta2.addForce(-1*F1) #Siente la fuerza en sentido contrario pero de la misma magnitud.
        #Cálculo energía potencial
        self.EpTotal += -G_sun * m2 / r_mag #Energía potencial por unidad de masa de la tierra. 

    def calculateAllForces(self,Planetas,r_arrays):
        self.EpTotal = 0.0 #Se reinicia la energía potencial
        #Se reinician las fuerzas de todos los planeta
        for i in range(0,N):
            Planetas[i].resetForce()
            Planetas[i].r = r_arrays[i]
            #Recorro por parejas, calculo la fuerza de cada pareja y se la sumo a los dos

        #N=2 (i=1,j=0)
        for i in range(0,N):
            for j in range(0,i): #Evita repetición. Solo se calcula (1,2) y no (2,1) porque la fuerza es la misma pero en sentido opuesto.
                self.calculateForceBetween(Planetas[i],Planetas[j])



def graphicPlanets(positionsBodies, velocityBodies, energy, t_array):
    plt.style.use('seaborn-v0_8')
    #(t,N,3)
    # Índices donde los valores son diferentes de 0
    # indices = np.nonzero(arr)
    # print(indices)  # (array([1, 3, 4]),)

    # Guardar trayectorias en un PDF
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # indices = np.nonzero(positionsBodies[:,0,0])
    # print(indices)
    # indices = np.nonzero(positionsBodies[:,0,1])
    # print(indices)
    # indices = np.nonzero(velocityBodies[:,0,0])
    # print(indices)
    # indices = np.nonzero(velocityBodies[:,0,1])
    # print(indices)


    x = positionsBodies[:, 1, 0]  # Componente x del cuerpo 1
    y = positionsBodies[:, 1, 1]  # Componente y del cuerpo 1
    ax1.plot(x, y, label=f'Tierra')  # Graficar la trayectoria
    ax1.scatter(0, 0, color='yellow', marker='o', s=80) #Sol
    ax1.scatter(x[-1], y[-1], color='blue', marker='o', s=80) #Tierra
    # Añadir etiquetas y leyenda
    ax1.set_xlabel('x [UA]')
    ax1.set_ylabel('y [UA]')
    ax1.set_title('Órbita tierra-sol')
    ax1.legend()
    ax1.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('verlet.pdf')
    print("verlet.pdf creado.")
    plt.close(fig1)

    # Guardar energías en otro PDF
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))  # 3 subplots

    # Energía Potencial
    ax2 = axes[0]
    energia_pot_inicial = energy[0, 0]
    energia_pot_promedio = np.mean(energy[:, 0])
    energia_pot = energy[:, 0]
    energia_pot_desv = np.std(energia_pot)
    ax2.plot(t_array[:-1], energia_pot, label=f'Energía Potencial\nInicial: {energia_pot_inicial:.7e}\nPromedio: {energia_pot_promedio:.7e}\nDesviación estándar: {energia_pot_desv:.2e}', color='blue')
    ax2.set_xlabel('Tiempo [yr]')
    ax2.set_ylabel(r'Energía [$UA^2/yr^2$]')
    ax2.set_title('Energía Potencial')
    ax2.legend()
    ax2.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))  # Notación científica

    # Energía Cinética
    ax3 = axes[1]
    energia_cin_inicial = energy[0, 1]
    energia_cin_promedio = np.mean(energy[:, 1])
    energia_cin = energy[:, 1]
    energia_cin_desv = np.std(energia_cin)
    ax3.plot(t_array[:-1], energia_cin, label=f'Energía Cinética\nInicial: {energia_cin_inicial:.7e}\nPromedio: {energia_cin_promedio:.7e}\nDesviación estándar: {energia_cin_desv:.2e}', color='orange')
    ax3.set_xlabel('Tiempo [yr]')
    ax3.set_ylabel(r'Energía [$UA^2/yr^2$]')
    ax3.set_title('Energía Cinética')
    ax3.legend()
    ax3.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))  # Notación científica
    print(energia_cin[0], energia_cin[int(len(energia_cin)/2)], energia_cin[-1])
    
    # Energía Total
    ax4 = axes[2]
    energia_total = energia_cin + energia_pot
    energia_total_inicial = energia_total[0]
    energia_total_promedio = np.mean(energia_total)
    ax4.scatter(t_array[:-1], energia_total, label=f'Energía Total\nInicial: {energia_total_inicial:.7e}\nPromedio: {energia_total_promedio:.7e}', color='green')
    ax4.set_xlabel('Tiempo [yr]')
    ax4.set_ylabel(r'Energía [$UA^2/yr^2$]')
    ax4.set_ylim(np.min(energia_total)-0.01,np.max(energia_total)+0.01)
    ax4.set_title('Energía Total')
    ax4.legend()
    ax4.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))  # Notación científica
    
    # Ajustar diseño y guardar
    plt.tight_layout()
    plt.savefig('energyVerlet.pdf')
    print("energyVerlet.pdf creado.")
    plt.close(fig2)




def verletAlgorithm(finalTime,dt,Planetas):

        Newton = dynamicManager()

        t_array = np.arange(0,finalTime + dt, dt)

        positionsBodies = np.zeros((len(t_array),N,3))
        velocityBodies = np.zeros((len(t_array),N,3))
        energy = np.zeros((len(t_array)-1,2))

        initialPosition = np.array([cuerpo.r for cuerpo in Planetas])
        initialVelocity = np.array([cuerpo.V for cuerpo in Planetas])
        masses = np.array([cuerpo.m for cuerpo in Planetas])    


        #Inicialización:
        #x_{-1} = x_0 - v0*dt
        position_verletm1 = initialPosition - initialVelocity*dt #x_{-1} #y_{-1} #z_{-1} para todos los cuerpos

        Newton.calculateAllForces(Planetas, initialPosition) #Cada planeta tiene su fuerza calculada.

        #a0 = a_0_cuerpo1 + a_0_cuerpo2 = {(a0x1,a0y1,a0z1),(a0x2,a0y2,a0z2)}
        a_0 = np.array([(cuerpo.F)/cuerpo.m for cuerpo in Planetas]) #Tripletas de aceleración para todos los cuerpos.

        #x_{n+1} = 2*x_n - x_{nx-1} + a_n*(dt**2)
        #(n=0) x_1 = 2*x_0 - x_{-1} + a_0*(dt**2)
        position_verlet_step_np1 = 2*initialPosition - position_verletm1 + a_0*(dt**2) #Se tienen tripletas de el primer paso de evolución.
        positionsBodies[0] = initialPosition #(len(t),N,3)
        positionsBodies[1] = position_verlet_step_np1


        #v_n = (x_{n+1} - x_{n-1})*(1/(2*dt))
        #v_1 = x_2 - x_0 (Aún no se puede calcular)
        velocityBodies[0] = initialVelocity #[(vx_0,vy_0,vz_0),(vx_1,vy_1,vz_1)]

        energy[0][0] = Newton.EpTotal
        energy[0][1] = (1/2)*np.sum(np.sum(initialVelocity**2, axis=1)) #No se incluye la multiplicación de la masa, pues debería ser la de la tierra. Es la energía cinética por unidad de masa.


        #Algoritmo
        for i in range(2,len(t_array)):
            #Ya tengo x0 y x1
            #x_{n+1} = 2*x_n - x_{n-1} + a_n*(dt**2)
            #(n=1) x_{2} = 2*x_1 - x_0 + a_1*(dt**2)
            #Evolución posición:
            position_verlet_step_nm1 = initialPosition
            position_verlet_step_n = position_verlet_step_np1
            Newton.calculateAllForces(Planetas, position_verlet_step_n) #Se encarga de actualizar posición y luego de calcular fuerzas.
            a_n = np.array([(cuerpo.F)/cuerpo.m for cuerpo in Planetas]) #Tripletas de aceleración para todos los cuerpos.
            position_verlet_step_np1 = 2*position_verlet_step_n - position_verlet_step_nm1 + a_n*(dt**2) #Se tienen tripletas de el primer paso de evolución.
            initialPosition = position_verlet_step_n    #(n=2) x_{3} = 2*x_2 - x_1 + a_2*(dt**2)
            positionsBodies[i] = position_verlet_step_np1

            #Evolución velocidad:

            #v_n = (x_{n+1} - x_{n-1})*(1/(2*dt))
            #(n=1) v_1 = (x_{2} - x_{0})*(1/(   2*dt))
            velocity = (position_verlet_step_np1 - position_verlet_step_nm1)/(2*dt)
            velocityBodies[i-1] = velocity


            energy[i-1][0] = Newton.EpTotal
            energy[i-1][1] = (1/2)*np.sum(np.sum(velocity**2, axis=1))

        return positionsBodies,velocityBodies,energy,t_array


    
def main():

        #Sistema Tierra-Sol:
        m0 = 1 #masa solar
        m1 =  M_earth*(1/solar_mass)
        m1 = m1.value
        x0_1 = 1.4710e11/a_unit
        vy0_1 = (3.0287e4/a_unit)*yrToSec


        #Proyecto 2:
        # m0 = 1
        # m1 = 5.97219e10*(1/solar_mass)
        # x0_1 = (4e12)/a_unit
        # vy0_1 = (500/a_unit)*yearToSeg
        # finalTime = 49.5 #Periodo es aproximadamente 49.5 años


        #---------------(x0,y0,z0,Vx0,Vy0,Vz0,m0,)
        Planeta0 = Cuerpo(0, 0, 0, 0, 0,  0, m0)
        Planeta1 = Cuerpo(x0_1, 0, 0, 0, vy0_1,  0,m1)
        Planetas = np.array((Planeta0, Planeta1))


        # print("=== Datos iniciales de los cuerpos cargados ===")
        # for i, p in enumerate(Planetas):
        #   print(f"Cuerpo {i:2d}: "
        #       f"r = ({p.r[0]: .6f}, {p.r[1]: .6f}, {p.r[2]: .6f}), "
        #       f"V = ({p.V[0]: .6f}, {p.V[1]: .6f}, {p.V[2]: .6f}), "
        #       f"m = {p.m:.6e}")
        # print("=============================================\n")

        #Configuración temporal
        finalTime = 1 # Un año
        dt= hour.to(yr) #Paso de tiempo de 1 hora

        positionsBodies,velocityBodies,energy,t_array = verletAlgorithm(finalTime,dt,Planetas)


        graphicPlanets(positionsBodies, velocityBodies, energy, t_array)





if __name__ == "__main__":
    main()

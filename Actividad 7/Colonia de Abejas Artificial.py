import matplotlib.pyplot as plt
import numpy as np
import random
import copy

def f(x,y):
    return x*np.exp(-x**2-y**2)

# Función para visualizar la superficie de la función objetivo
def plot_surface():
    plotN = 200
    x = np.linspace(-1,1, plotN)
    y = np.linspace(-1,1, plotN)
    x, y = np.meshgrid(x,y)
    z = f(x,y)

    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.plot_surface(x, y, z, cmap='jet', shade=False)
    plt.show()

# Función para visualizar las curvas de nivel de la función objetivo
def plot_contour():
    plotN = 200
    x = np.linspace(-1,1, plotN)
    y = np.linspace(-1,1, plotN)
    x, y = np.meshgrid(x,y)
    z = f(x,y)

    plt.contour(x,y,z)
    plt.show()

# Función para la selección de abejas
def Selection(aptitud):
    aptitud_total = sum(aptitud)
    N = len(aptitud)
    r=random.random()
    P_sum = 0

    for i in range(0,N):
        p_sum = P_sum + aptitud[i]/ aptitud_total

        if P_sum >= r:
            n=copy.copy(i)
            return n
    n = N-1
    return n

# Algoritmo de Colonia de Abejas Artificial (ABC)
def ABC():
    dimension = 2
    x_lower = np.array([-2 , -2]) # cota minima
    x_upper = np.array([2, 2])   # cota maxima

    N = 50 # Tamaño de la colonia
    L = 15 # Número de intentos para explotar el alimento
    Pf = 30 # Número de abejas obreras
    Po = N-Pf # Número de abejas observadoras

    # Crear vectores de población, aptitud, fitness y n
    x_test = np.zeros((Pf,dimension))
    l = np.zeros((Pf,1))
    fitness = np.zeros((Pf,1))
    aptitud = np.zeros((Pf,1))

    # Inicializar población y fitness
    for i in range(0,Pf):
        x_test[i,:] = x_lower + (x_upper - x_lower)*np.array([random.random(), random.random()])
        fitness[i] = f(x_test[i,0],x_test[i,1])

    # Iterar ABC
    for g in range(1, 300):
        # Abejas obreras
        for i in range(0,Pf):
            k=copy.copy(i)
            while k==i:
                k= random.randint(0,Pf-1)
            j = random.randint(0,dimension-1)
            phi = 2*random.random()-1
            v= copy.copy(x_test[i,:])
            v[j] = x_test[i,j] + phi*(x_test[i,j]-x_test[k,j])
            fv = f(v[0],v[1])

            if fv < fitness[i]:
                x_test[i,:] = v
                fitness[i] = fv
                l[i] = 0
            else:
                l[i] = l[i]+1

            if fitness[i]>=0:
                aptitud[i] = 1/(1+fitness[i])
            else:
                aptitud[i] = 1+ abs(fitness[i])

        # Abejas observadoras
        for i in range(0,Po):
            m = Selection(aptitud)
            k= copy.copy(m)
            while k==m:
                k = random.randint(0,Pf-1)
            j = random.randint(0,dimension-1)
            phi = 2*random.random()-1
            v = x_test[m,:]
            v[j] = x_test[m,j] + phi*(x_test[m,j]-x_test[k,j])
            fv = f(v[0],v[1])

            if fv < fitness[m]:
                x_test[m,:] = v
                fitness[m] = fv
                l[m] = 0
            else:
                l[m] = l[i]+1

        # Abejas exploradoras
        for i in range(0,Pf):
            if l[i] > L:
                x_test[i,:] = x_lower + (x_upper - x_lower)*np.array([random.random(), random.random()])
                fitness[i] = f(x_test[i,0],x_test[i,1])
                l[i]=0

    best = min(fitness)
    x_best = np.where(fitness == best)
    print("Valores que optimizan la función =",x_test[x_best[0][0],0],  x_test[x_best[0][0],1])
    print("Minimo en =", f(x_test[x_best[0][0],0],  x_test[x_best[0][0],1]))

# Visualizar la superficie de la función objetivo
plot_surface()

# Visualizar las curvas de nivel de la función objetivo
plot_contour()

# Ejecutar el algoritmo ABC
ABC()
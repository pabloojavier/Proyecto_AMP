# -*- coding: utf-8 -*-
import tsplib95
import matplotlib.pyplot as plt
import random
import time
import sys
import array
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from visualizar import animacion

Infinity = 9999999
dist = []
class Instancia:
    def __init__(self, graficar_ruta, instance):
        self.graficar_ruta = graficar_ruta
        self.coord_x = []
        self.coord_y = []
        self.problem = tsplib95.load(instance)
        self.info = self.problem.as_keyword_dict()
        self.n = len(self.problem.get_graph())
        if self.info['EDGE_WEIGHT_TYPE'] == 'EUC_2D': # se puede graficar la ruta
            for i in range(1, self.n + 1):
                x, y = self.info['NODE_COORD_SECTION'][i]
                self.coord_x.append(x)
                self.coord_y.append(y)
        else:
            self.graficar_ruta = False

    def generarMatriz(self):
        inicio = list(self.problem.get_nodes())[0]
        global dist
        dist = [[Infinity for i in range(self.n)] for k in range(self.n)]
        if inicio == 0:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        u = i, j
                        dist[i][j] = self.problem.get_weight(*u)
        else:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        u = i + 1, j + 1
                        dist[i][j] = self.problem.get_weight(*u)

# distancia entre la ciudad i y j
def distancia(i, j):
    return dist[i][j]

# Costo de la ruta
def costoTotal(ciudad):
    suma = 0
    i = 0
    while i < len(ciudad) - 1:
        suma += distancia(ciudad[i], ciudad[i + 1])
        i += 1
    suma += distancia(ciudad[-1], ciudad[0])
    return suma,

# heurística del vecino más cercano
def vecinoMasCercano(n):
    desde = random.randrange(0, n)
    if random.uniform(0, 1) < 0.3:
        actual = desde
        ciudad = []
        ciudad.append(desde)
        seleccionada = [False] * n
        seleccionada[actual] = True

        while len(ciudad) < n:
            min = Infinity
            for candidata in range(n):
                if seleccionada[candidata] == False and candidata != actual:
                    costo = distancia(actual, candidata)
                    if costo < min:
                        min = costo
                        siguiente = candidata

            ciudad.append(siguiente)
            seleccionada[siguiente] = True
            actual = siguiente
    else:
        ciudad = [i for i in range(0, n)]
        random.shuffle(ciudad)
    return ciudad

# Búsqueda local 2-opt
def DosOpt(ciudad):
    n = len(ciudad)
    flag = True
    contar = 0
    actual = 0
    k = random.randint(0, n - 1)
    ciudad = ciudad[k:] + ciudad[:k] # si usan numpy cambiar
    #ciudad = np.hstack((ciudad[k:], ciudad[:k])) # con numpy
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            nuevoCosto = distancia(ciudad[i], ciudad[j]) + distancia(ciudad[i + 1], ciudad[j + 1]) - distancia(ciudad[i], ciudad[i + 1]) - distancia(ciudad[j], ciudad[j + 1])
            if nuevoCosto < actual:
                actual = nuevoCosto
                min_i, min_j = i, j
                # Al primer cambio se sale
                contar += 1
                if contar == 1 :
                    flag = False

        if flag == False:
            break

    # Actualiza la subruta se encontró
    if contar < 0:
        ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]

# perturbación: se escogen dos ciudades aleatorias y las intercambia
def perturbation(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i == j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    # intercambio
    temp = ciudad[i]
    ciudad[i] = ciudad[j]
    ciudad[j] = temp

# perturbación: se escoge una ciudad aleatoria y se intercambia con la ciudad siguiente en la ruta
def perturbation3(ciudad):
    j = 0
    n = len(ciudad)
    i = random.randint(0, n - 1)
    if i == n - 1:
        j = 0
    else:
        j = i + 1
    # intercambio
    temp = ciudad[i]
    ciudad[i] = ciudad[j]
    ciudad[j] = temp

# perturbación 2: dos puntos aleatorios e invierte las ciudades entremedio
def perturbation2(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i >= j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    ciudad[i : j] = ciudad[i : j][::-1]

def mutacion(ciudad):
    #perturbation(ciudad)
    # value = random.uniform(0, 1)
    # if value < 0.3:
    #     perturbation(ciudad)
    # elif value >= 0.3 and value < 0.6:
    #     perturbation2(ciudad)
    # elif value >= 0.6 and value < 0.85:
    perturbation2(ciudad)
    # else:
    #     DosOpt(ciudad)

    return ciudad,

def GA_simple(ins, semilla):
    # Parámetros
    poblacion = 50  # tamaño población
    iterMax = 200   # número de generaciones
    CXPB = 0.9      # probabilidad de cruzamiento
    MUTPB = 0.1     # probabilidad de mutación
    nTorneo = 4
    n = ins.n
    random.seed(semilla)

    # Definir individuos y su fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin) # con numpy
    toolbox = base.Toolbox()

    # Población inicial
    toolbox.register("indices", vecinoMasCercano, n)

    # Formato de los individuos y población
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Función objetivo
    toolbox.register("evaluate", costoTotal)

    # Selección
    toolbox.register("select", tools.selTournament, tournsize=nTorneo)

    # Cruzamiento
    toolbox.register("mate", tools.cxOrdered)

    # Mutación
    toolbox.register("mutate", mutacion)

    # Estadísticas
    pop = toolbox.population(n=poblacion)
    hof = tools.HallOfFame(1, similar=np.array_equal) # con numpy

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evolucionando
    inicioTiempo = time.time()
    result, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, iterMax, stats=stats, halloffame=hof)
    finTiempo = time.time()
    tiempo = finTiempo - inicioTiempo

    # Generando la estadísticas
    minimo, promedio = log.select("min", "avg")
    best_individual = tools.selBest(result, k=1)[0]
    print('Costo  : %d' % costoTotal(best_individual)[0])
    print("Tiempo : %f" % tiempo)
    if ins.graficar_ruta:
        graficar_soluciones(minimo, promedio)

def GA_avanzado(ins, semilla):
    # Parámetros
    poblacion = 50 # tamaño población
    iterMax = 200   # número de generaciones
    CXPB = 0.9      # probabilidad de cruzamiento
    MUTPB = 0.1     # probabilidad de mutación
    nTorneo = 4
    n = ins.n
    random.seed(semilla)
    # Definir individuos y su fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin) # sin numpy
    #creator.create("Individual", np.ndarray, fitness=creator.FitnessMin) # con numpy
    toolbox = base.Toolbox()

    # Población inicial
    #toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("indices", vecinoMasCercano, n)

    # Formato de los individuos y población
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Función objetivo
    toolbox.register("evaluate", costoTotal)

    # Selección
    toolbox.register("select", tools.selTournament, tournsize=nTorneo)

    # Cruzamiento
    #toolbox.register("mate", tools.cxPartialyMatched)
    #toolbox.register("mate", tools.cxUniformPartialyMatched)
    toolbox.register("mate", tools.cxOrdered)

    # Mutación
    #toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("mutate", mutacion)

    # Estadísticas
    pop = toolbox.population(n=poblacion)
    hof = tools.HallOfFame(1) # sin numpy
    #hof = tools.HallOfFame(1, similar=np.array_equal) # con numpy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    log = tools.Logbook()
    log.header = "gen", "evals", "std", "min", "avg", "max"

    # Evolución
    inicioTiempo = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    g = 0
    lista_soluciones = []
    lista_costos = []
    record = stats.compile(pop)
    log.record(gen=g, evals=len(pop), **record)
    print(log[-1]["gen"], log[-1]["avg"], log[-1]["min"])
    while g < iterMax:
        g = g + 1
        # Selección
        offspring = toolbox.select(pop, len(pop))

        # Cruzamiento
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values # eliminar costo para recalcular luego
                del child2.fitness.values

        # Mutación
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # # Memetic
        # for i in offspring:
        #     DosOpt(i)
        #     del i.fitness.values

        # Evaluación
        ind_sincalcular = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, ind_sincalcular)
        for ind, fit in zip(ind_sincalcular, fitnesses):
            ind.fitness.values = fit

        # Reemplazamiento
        pop[:] = offspring
        hof.update(offspring)
        record = stats.compile(offspring)
        log.record(gen=g, evals=len(offspring), **record)
        print(log[-1]["gen"], log[-1]["avg"], log[-1]["min"])

        top = tools.selBest(offspring, k=1)
        lista_costos.append(int(log[-1]["min"]))
        lista_soluciones.append(top[0])

    finTiempo = time.time()
    tiempo = finTiempo - inicioTiempo
    minimo, promedio = log.select("min", "avg")

    print('Costo  : %d' % min(lista_costos))
    print("Tiempo : %f" % tiempo)
    # Graficar rutas y animación
    if ins.graficar_ruta:
        ver = animacion(lista_soluciones, ins.coord_x, ins.coord_y, lista_costos)
        ver.animacionRutas()
        graficar_soluciones(minimo, promedio)

def graficar_soluciones(minimo, promedio):
    plots = plt.plot(minimo,'c-', promedio, 'b-')
    plt.legend(plots, ('Costo Mínimo', 'Costo Promedio'), frameon=True)
    plt.ylabel('Costo')
    plt.xlabel('Generaciones')
    plt.title("Generaciones vs Costo - TSP")
    plt.xlim((0, len(minimo)))
    plt.show()

def main():
    if len(sys.argv) < 5:
        print("Uso: python3 ga.py grafica[No(0), Si(1)] nombreInstancia semilla version")
        return
    dot = int(sys.argv[1]) # 0 no graficar; 1 graficar
    instance = sys.argv[2] # ruta/nombre instancias
    semilla = int(sys.argv[3]) # semilla aleatoria
    version = int(sys.argv[4]) # version GA: simple 0; avanzado 1
    ins = Instancia(dot, instance)
    ins.generarMatriz()
    if version == 0:
        GA_simple(ins, semilla)
    else:
        GA_avanzado(ins, semilla)

if __name__ == "__main__":
    main()

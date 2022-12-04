import tsplib95
import matplotlib.pyplot as plt
import random
import time
import sys
import array
import numpy as np
import pandas as pd
import copy
import math
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def read_data(ins):
    dist = pd.read_csv(f"Instancias/Dist/{ins}.txt",sep="\t",header=None).to_numpy()
    try:
        aux = pd.read_csv(f"Instancias/Coords/{ins}.txt",sep="\t",index_col=0).to_dict()
        coords = {i-1:{"x":aux["x-coor"][i],"y":aux["y-coor"][i]}  for i in range(1,len(aux["x-coor"])+1)}
        #coords = {i:(aux["x-coor"][i],aux["y-coor"][i]) for i in range(1,len(aux["x-coor"]))}
    except:
        coords = None
    
    return dist,coords

def costo(vector_sol):
    #    valores = [0 for i in range(m)]
    # for i in range(m):
    #     valores[i] = costo_parcial(vector_sol[i])
        # valores[i] = distancia[0][vector_sol[i][0]] + distancia[0][vector_sol[i][-1]] #Se suma el valor desde y hasta el nodo inicial
        
        # for j in range(len(vector_sol[i])-1):
        #     valores[i] += distancia[vector_sol[i][j]][vector_sol[i][j+1]]
    if len(vector_sol)<3:
        print("hola")
    valores = [costo_parcial(vector_sol[i]) for i in range(m)]
    if tipo == "minsum":
        return sum(valores),
    else: 
        return max(valores),

def costo_parcial(ruta):
    try:
        valor = distancia[0][ruta[0]] + distancia[0][ruta[-1]]
    except:
        raise
    for i in range(len(ruta)-1):
        valor += distancia[ruta[i]][ruta[i+1]]
    return valor

def m_vecino_mas_cercano_V2():
    """
    Vecino más cercano adaptado para m rutas, partiendo desde cualquiera

    """
    nodos = [i for i in range(1,n)]
    visitados = []
    solucion = []

    while len(solucion)<m:
        nodo = random.randint(1,n-1)
        if nodo not in visitados:
            solucion.append([nodo])
            visitados.append(nodo)
    # print(visitados)
    pendientes = [i for i in nodos if i not in visitados]
    # os.system("clear")
    # print("sol actual:",solucion)
    # print("pendientes",pendientes)
    while len(pendientes) != 0:
        for i in range(m):
            costos_actual = {(solucion[i][-1],j):distancia[solucion[i][-1]][j] for j in pendientes}
            mejores = sorted(costos_actual.items(),key=lambda item:item[1])
            mejor_nodo = mejores[0][0][1]
            solucion[i].append(mejor_nodo)
            pendientes.remove(mejor_nodo)
            if len(pendientes)==0:
                break
    
    return solucion

def m_vecino_mas_cercano():
    """
    Vecino más cercano adaptado para m rutas, partiendo desde el mas cercano a 0

    """
    costos_depot = {(0,i):distancia[0][i] for i in range(1,n)}
    mejores = sorted(costos_depot.items(),key=lambda item:item[1])[:m] #Mejores m nodos desde deposito
    nodos = [i for i in range(1,n)]
    solucion = [[mejores[i][0][1]] for i in range(m)]
    visitados = [nodo[0] for nodo in solucion]
    pendientes = [i for i in nodos if i not in visitados]
    # os.system("clear")
    # print("sol actual:",solucion)
    # print("pendientes",pendientes)
    while len(pendientes) != 0:
        for i in range(m):
            costos_actual = {(solucion[i][-1],j):distancia[solucion[i][-1]][j] for j in pendientes}
            mejores = sorted(costos_actual.items(),key=lambda item:item[1])
            mejor_nodo = mejores[0][0][1]
            solucion[i].append(mejor_nodo)
            pendientes.remove(mejor_nodo)
            if len(pendientes)==0:
                break
    
    return solucion

def solucion_paper():
    """
    En cada ruta se agregan los m nodos más cercanos.
    Luego con cada nodo restante se analiza en que ruta
    es mejor añadirlo.
    """
    costos_depot = {(0,i):distancia[0][i] for i in range(1,n)}
    mejores = sorted(costos_depot.items(),key=lambda item:item[1])
    solucion = [[mejores[i][0][1]] for i in range(m)]
    
    nodos = [i for i in range(1,n)]
    visitados = [nodo[0] for nodo in solucion]
    pendientes = [i for i in nodos if i not in visitados]
    for i in pendientes:
        length = [costo_parcial(solucion[v]) for v in range(m)]
        avg_length = np.mean(length)
        if tipo == "minmax":
            v_b_list = [(distancia[solucion[v][-1]][i] *(1+(length[v]-avg_length)/(avg_length)),v) for v in range(m) ]
            v_b = min(v_b_list,key=lambda x:x[0])[1] 
        
        elif tipo == "minsum":
            v_b_list = [(distancia[solucion[v][-1]][i] + distancia[i][0]-distancia[solucion[v][-1]][0],v) for v in range(m)]
            v_b = min(v_b_list,key=lambda x:x[0])[1] 
    

        solucion[v_b].append(i)

    return solucion

def random_solution():
    tamanos = [1] * m
    for i in range(n-m-1) :
        tamanos[random.randint(0, n) % m] += 1
    aux = [i for i in range(1,n)]

    random.shuffle(aux)

    cum_tamanos = np.cumsum(tamanos)
    solucion = [[] for i in range(m)]
    solucion[0] = aux[:tamanos[0]].copy()
    for i in range(1,m):
        solucion[i] = aux[cum_tamanos[i-1]:cum_tamanos[i]]

    return solucion

def poblacion_inicial():
    random1 = random.random()
    if random1 < 1/6:
        solucion = m_vecino_mas_cercano_V2()
    elif random1 < 2/6:
        solucion = m_vecino_mas_cercano()
    elif random1 < 4/5:
        solucion = solucion_paper()
    else:
        solucion = random_solution()
    return solucion

def perturbacion1(sol):
    """
    Mueve un nodo aleatorio de una ruta aleatoria
    a una posición aleatoria de otra ruta aleatoria.
    Si la ruta es la misma el nodo se coloca en otra posición.
    """
    v = random.randint(0,len(sol)-1)
    while len(sol[v])<=1:
        v = random.randint(0,len(sol)-1)

    i = random.choice(sol[v])
    t = random.randint(0,len(sol)-1)
    # while t == v:
    #     t = random.randint(0,len(sol)-1)
    j = random.randint(0,len(sol[t]))
    while t == v and j == i:
        j = random.randint(0,len(sol[t]))
    sol[v].remove(i)
    sol[t].insert(j,i)
    return sol

def perturbacion2(solucion):
    """
    Perturba entre 1 y m rutas.
    En cada perturbación se busca una ruta aleatoria que tenga
    más de 1 nodo.
    Perturba intercambiando con su sucesor.
    """
    k = random.randint(1,m-1)
    #random.shuffle(sol[k])
    for i in range(k):
        ruta = random.randint(0,m-1)
        if len(solucion[ruta])!=1:
                nodo = random.randint(0,len(solucion[ruta])-1)
                solucion[ruta][nodo:nodo+2] = solucion[ruta][nodo:nodo+2][::-1]

def one_point_move(solucion):
    """
    Inserta un nodo en todas las posibles posiciones de otras rutas
    Se queda con el mejor vecino
    """
    if tipo == "minsum":
        v = random.randint(0,m-1)
    else:
        #v = max([(len(solucion[i]),i) for i in range(m)] , key=lambda x:x[0] )[1]
        v = max([(costo_parcial(solucion[i]),i) for i in range(m)],key=lambda x:x[0])[1]
    mejor_costo = costo(solucion)
    
    ruta_original = copy.deepcopy(solucion)
    candidato = copy.deepcopy(solucion)
    for i in solucion[v]:
        for v2 in range(m):
            if v2 != v:
                for j in range(len(solucion[v2])+1):
                    if len(candidato[v])>1:
                        candidato[v].remove(i)
                        candidato[v2].insert(j,i)
                        costo_cambio = costo(candidato)
                        if costo_cambio < mejor_costo:
                            mejor_costo = costo_cambio
                            solucion = copy.deepcopy(candidato)
                            return 
                        candidato = copy.deepcopy(ruta_original)

def two_opt_move(solucion):
    #print("inicial: ",solucion,costo(solucion,tipo),"\n")
    for ciudad in solucion:
        #ciudad = solucion[random.randint(0,m-1)]
        actual = 0
        n = len(ciudad)
        flag = True
        contar = 0
        # k = random.randint(0, len(ciudad) - 1)
        # ciudad = ciudad[k:] + ciudad[:k]
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                nuevoCosto = distancia[ciudad[i]][ciudad[j]] + distancia[ciudad[i + 1]][ciudad[j + 1]] - distancia[ciudad[i]][ciudad[i + 1]] - distancia[ciudad[j]][ciudad[j + 1]]
                if nuevoCosto < actual:
                    actual = nuevoCosto
                    min_i, min_j = i, j
                    # Al primer cambio se sale
                    #contar += 1
                    #if contar == 1 :
                        #print("entre",actual,min_i,min_j,ciudad)
                     #   flag = False

            #if flag == False:
             #   break

        # Actualiza la subruta se encontró
        if actual < 0:
            ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]

def mutacion(solucion):
    random2 = random.random()
    if random2 < 1/4:
        perturbacion1(solucion)
    elif random2 < 2/4:
        perturbacion2(solucion)
    elif random2 < 3/4:
        one_point_move(solucion)
    else:
        two_opt_move(solucion)

def CXordered(individuo1, individuo2):
    ind1 = individuo1[random.randint(0,m-1)]
    ind2 = individuo2[random.randint(0,m-1)]
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a
    #print(ind1,ind2,size,a,b)

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        #print(holes2,temp2[(i+b+1)%size])
        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2

def cxPartialyMatched(individuo1, individuo2):
    return individuo1, individuo2
    al1 = random.randint(0,m-1)
    al2 = random.randint(0,m-1)

    ind1 = copy.deepcopy(individuo1[al1])
    ind2 = copy.deepcopy(individuo2[al2])

    size = min(len(ind1), len(ind2))

    # for i in range(m):
    #     if size<3:
    #         break
    #     else:
    #         ind1 = individuo1[random.randint(0,m-1)] 
    #         ind2 = individuo2[random.randint(0,m-1)]     
    #         size = min(len(ind1), len(ind2))
    # if size<3:
    #     return individuo1,individuo2

    p1, p2 = [0] * size, [0] * size 
    print(p1,ind1)
    print(p2,ind2)
    # Initialize the position of each indices in the individuals
    for i in range(size):
        #print(p1[ind1[i]])
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

def GA(semilla):
    # Parámetros
    poblacion = 50  # tamaño población
    iterMax = 200   # número de generaciones
    CXPB = 0.9      # probabilidad de cruzamiento
    MUTPB = 0.2     # probabilidad de mutación
    nTorneo = 4
    random.seed(semilla)

    # Definir individuos y su fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual",list, typecode='i' , fitness=creator.FitnessMin) 
    toolbox = base.Toolbox()

    # Población inicial
    toolbox.register("indices", poblacion_inicial)
    #toolbox.register("cromo",poblacion_inicial)

    # Formato de los individuos y población
    toolbox.register("individual", tools.initIterate,creator.Individual,toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Función objetivo
    toolbox.register("evaluate", costo)

    # Selección
    toolbox.register("select", tools.selTournament, tournsize=nTorneo)

    # Cruzamiento
    toolbox.register("mate", cxPartialyMatched)

    # Mutación
    toolbox.register("mutate", mutacion)

    # Estadísticas
    pop = toolbox.population(n=poblacion)
    # for i in pop:
    #     print(i.fitness.values)
    #     exit(0)
    hof = tools.HallOfFame(1) # con numpy

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evolucionando


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
    #print(log[-1]["gen"], log[-1]["avg"], log[-1]["min"])
    #while g < iterMax:
    while time.time()- inicioTiempo < n:
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
        #print(log[-1]["gen"], log[-1]["avg"], log[-1]["min"])
        print(f"{ins} {m} {log[-1]['gen']} {round(log[-1]['min'],2)} {round(time.time()- inicioTiempo,2)}",end="\r")
        top = tools.selBest(offspring, k=1)
        lista_costos.append(int(log[-1]["min"]))
        lista_soluciones.append(top[0])

    finTiempo = time.time()
    tiempo = finTiempo - inicioTiempo

    # Generando la estadísticas
    minimo, promedio = log.select("min", "avg")

    #print('Costo  : %d' % min(lista_costos))
    #print("Tiempo : %f" % tiempo)
    print()
    # if ins.graficar_ruta:
    #     graficar_soluciones(minimo, promedio)


instancias = ["51"]#["11a","11b","12a","12b","16","51","100","128","150"]#["100","128","150"]
ms = {"11a":[3],"11b":[3],"12a":[3],"12b":[3],"16":[3],"128":[10,15,30],"51":[3,5,10],"100":[3,5,10,20],"150":[3,5,10,20,30]}
instancia = "11a"
tipos = ["minmax","minsum"]#[::-1]

for tipo in tipos:
    for ins in instancias:
        distancia,coords = read_data(ins)
        n = len(distancia)
        for ma in ms[ins]:
            m = ma
            for seed in range(1):
                GA(seed)
                #exit(0)
    print("")
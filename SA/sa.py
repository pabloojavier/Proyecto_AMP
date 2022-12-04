# -*- coding: utf-8 -*-
import tsplib95
import matplotlib.pyplot as plt
import random
import time
import sys
import math
import numpy as np
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
        # print(ciudad[i], ciudad[i +1])
        suma += distancia(ciudad[i], ciudad[i + 1])
        i += 1
    suma += distancia(ciudad[-1], ciudad[0])
    return suma

# solución inicial completamente aleatoria
def solucionInicialAleatoria(n):
    ciudad = [i for i in range(0, n)]
    random.shuffle(ciudad)
    return ciudad

# heurística del vecino más cercano
def vecinoMasCercano(n, desde):
    actual = desde
    ciudad = []
    ciudad.append(desde)
    seleccionada = [False] * n
    seleccionada[actual] = True
    # print(seleccionada)
    while len(ciudad) < n:
        min = 9999999
        for candidata in range(n):
            if seleccionada[candidata] == False and candidata != actual:
                costo = distancia(actual, candidata)
                if costo < min:
                    min = costo
                    siguiente = candidata

        ciudad.append(siguiente)
        seleccionada[siguiente] = True
        actual = siguiente
    print(ciudad)
    print(costoTotal(ciudad))
    return ciudad

def DosOpt(ciudad):
    actual = 0
    n = len(ciudad)
    flag = True
    contar = 0
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
    if actual < 0:
        ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]

def invertir(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i >= j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    ciudad[i : j] = ciudad[i : j][::-1]

def SA(ins, semilla):
    alfa = 0.9995
    temperatura = 10
    temperatura_final = 1e-8
    max_vecinos = 3
    n = ins.n
    inicioTiempo = time.time()
    random.seed(semilla)
    # Solución inicial
    s = vecinoMasCercano(n, 0)
    #s = np.array(vecinoMasCercano(n, 0))
    #s = solucionInicialAleatoria(n)
    #s = np.array(solucionInicialAleatoria(n))
    s_mejor = s.copy()
    costoActual = costoTotal(s_mejor)
    costoMejor = costoActual
    iteracion = 0
    iterMax = 10000

    # lista_graficar = []
    # lista_soluciones = []
    # lista_costos = []
    # lista_costosMejores = []
    # lista_costos.append(costoMejor)
    # lista_costosMejores.append(costoMejor)
    # lista_soluciones.append(s_mejor)
    # tupla = costoMejor, temperatura
    # lista_graficar.append(tupla)
    while temperatura >= temperatura_final and iteracion < iterMax:
        for i in range(max_vecinos):
            s_candidato = s.copy()
            # Generar vecinos
            invertir(s_candidato)
            # DosOpt(s_candidato)
            costoCandidato = costoTotal(s_candidato)
            # Aceptar soluciones
            if costoCandidato < costoActual:
                costoActual, s = costoCandidato, s_candidato.copy()
                if costoCandidato < costoMejor:
                    costoMejor, s_mejor = costoCandidato, s_candidato.copy()
                    print(iteracion, temperatura, costoMejor)
            elif random.uniform(0, 1) < math.exp(-abs(costoCandidato - costoActual) / temperatura):
                costoActual, s = costoCandidato, s_candidato.copy()
        # Información
        # tupla = costoActual, temperatura
        # lista_graficar.append(tupla)
        # lista_costos.append(costoActual)
        # lista_costosMejores.append(costoMejor)
        # lista_soluciones.append(s)

        temperatura *= alfa
        iteracion += 1

    finTiempo = time.time()
    tiempo = finTiempo - inicioTiempo
    print("tiempo: ", tiempo, " segundos")
    print(s_mejor)
    print(costoTotal(s_mejor))
    print(ins.graficar_ruta)
    if ins.graficar_ruta:
        lista_soluciones.append(s_mejor)
        lista_costos.append(costoMejor)
        ver = animacion(lista_soluciones, ins.coord_x, ins.coord_y, lista_costos)
        ver.animacionRutas()
        graficar_soluciones(lista_graficar)

def graficar_soluciones(lista_graficar):
    t = [i for i in range(len(lista_graficar))]

    data1 = [i for i,j in lista_graficar]
    data2 = [j for i,j in lista_graficar]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Costo', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # agregar segundo eje

    color = 'tab:blue'
    ax2.set_ylabel('Temperatura', color=color)
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Iteraciones vs Costo / Temperatura - TSP")
    plt.xlim((0, len(lista_graficar)))

    fig.tight_layout()

    plt.show()

def main():
    # if len(sys.argv) < 4:
    #     print("Uso: python3 sa.py grafica[No(0), Si(1)] nombreInstancia semilla")
    #     return
    dot = 0#int(sys.argv[1]) # 0 no graficar; 1 graficar
    instance = "SA/instancias/bays29.tsp"#sys.argv[2] # ruta/nombre instancias
    semilla = 0#int(sys.argv[3]) # semilla aleatoria
    ins = Instancia(dot, instance)
    ins.generarMatriz()
    SA(ins, semilla)

if __name__ == "__main__":
    main()

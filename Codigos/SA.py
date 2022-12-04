import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math
import os
import copy 
import math
import numpy as np

def read_data(ins):
    dist = pd.read_csv(f"Instancias/Dist/{ins}.txt",sep="\t",header=None).to_numpy()
    try:
        aux = pd.read_csv(f"Instancias/Coords/{ins}.txt",sep="\t",index_col=0).to_dict()
        coords = {i-1:{"x":aux["x-coor"][i],"y":aux["y-coor"][i]}  for i in range(1,len(aux["x-coor"])+1)}
        #coords = {i:(aux["x-coor"][i],aux["y-coor"][i]) for i in range(1,len(aux["x-coor"]))}
    except:
        coords = None
    
    return dist,coords
    
def checker(solucion):
    ciudades = [0]
    if len(solucion)!=m: #Si es que no tiene m rutas
        return False

    for ruta in solucion: 
        if len(ruta)<1:
            return False #Todas las rutas deben visitar al menos un nodo
        for nodo in ruta:
            if nodo == 0:
                return False #El primer nodo no es parte de la representación
            ciudades.append(nodo)
    
    if len(set(ciudades))!= n: #Todos los nodos deben ser visitados
        return False

    return True

def costo(vector_sol,obj):
    #    valores = [0 for i in range(m)]
    # for i in range(m):
    #     valores[i] = costo_parcial(vector_sol[i])
        # valores[i] = distancia[0][vector_sol[i][0]] + distancia[0][vector_sol[i][-1]] #Se suma el valor desde y hasta el nodo inicial
        
        # for j in range(len(vector_sol[i])-1):
        #     valores[i] += distancia[vector_sol[i][j]][vector_sol[i][j+1]]
    if len(vector_sol)<3:
        print("hola")
    valores = [costo_parcial(vector_sol[i]) for i in range(m)]
    if obj == "minsum":
        return sum(valores)
    else: 
        return max(valores)

def costo_parcial(ruta):
    try:
        valor = distancia[0][ruta[0]] + distancia[0][ruta[-1]]
    except:
        raise
    for i in range(len(ruta)-1):
        valor += distancia[ruta[i]][ruta[i+1]]
    return valor

def graficar(solucion,obj="minmax"):
    colores = {0:"blue",1:"red",2:"green",3:"yellow",4:"black",5:"purple"}
    for i in range(len(solucion)):
        for j in range(len(solucion[i])):
            x = coords[solucion[i][j]]["x"]
            y = coords[solucion[i][j]]["y"]
            plt.scatter(x,y,c = colores[i],s=5)
            plt.annotate(f"{solucion[i][j]}",(x,y))

    for i in range(len(solucion)):    
        for j in range(len(solucion[i])-1):
            x = [coords[solucion[i][j]]["x"],coords[solucion[i][j+1]]["x"]]
            y = [coords[solucion[i][j]]["y"],coords[solucion[i][j+1]]["y"]]
            plt.plot(x,y,c = colores[i])

    for i in range(len(solucion)):
        xi = [coords[0]["x"],coords[solucion[i][0] ]["x"]]
        yi = [coords[0]["y"],coords[solucion[i][0] ]["y"]]
        xf = [coords[solucion[i][-1]]["x"],coords[0]["x"]]
        yf = [coords[solucion[i][-1]]["y"],coords[0]["y"]]
        plt.plot(xi,yi , c = colores[i])
        plt.plot(xf,yf , c = colores[i])
    

    plt.title(f"{obj}: {'%.2f'%costo(solucion,obj)}\n{solucion}")
    #plt.savefig(f"{ins}_{m}")
    plt.show()

def graficar_temperatura_costos(costos,temperaturas):
    iteraciones = [i for i in range(len(costos))]

    fig, ax1 = plt.subplots()
    blue = 'tab:blue'
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Costos', color=blue)
    ax1.plot(iteraciones, costos, color=blue)
    ax1.tick_params(axis='y', labelcolor=blue)

    ax2 = ax1.twinx() 
    red = 'tab:red'
    ax2.set_ylabel('Temperatura', color=red)  
    ax2.plot(iteraciones, temperaturas, color=red)
    ax2.tick_params(axis='y', labelcolor=red)

    fig.tight_layout()  
    
    plt.show()

def m_vecino_mas_cercano():
    """
    Vecino más cercano adaptado para m rutas
    """
    costos_depot = {(0,i):distancia[0][i] for i in range(1,n)}
    mejores = sorted(costos_depot.items(),key=lambda item:item[1])[:m] #Mejores m nodos desde deposito
    solucion = [[mejores[i][0][1]] for i in range(m)]
    nodos = [i for i in range(1,n)]
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

def random_solution(n,m):
    random.seed(0)
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

def generar_solucion():
    # mayor = "mio" if costo(solucion_paper(),tipo) > costo(m_vecino_mas_cercano(),tipo) else "paper"
    # print("{:<5}{:<3}{:<11}{:<15}{:<7}{:<6}".format(ins,m,"%.2f"%costo(solucion_paper(),tipo),"%.3f"%costo(m_vecino_mas_cercano(),tipo),mayor,tipo))

    #solucion = solucion_paper()
    #solucion = random_solution()
    solucion = m_vecino_mas_cercano()

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
    mejor_costo = costo(solucion,tipo)
    
    ruta_original = copy.deepcopy(solucion)
    candidato = copy.deepcopy(solucion)
    for i in solucion[v]:
        for v2 in range(m):
            if v2 != v:
                for j in range(len(solucion[v2])+1):
                    if len(candidato[v])>1:
                        candidato[v].remove(i)
                        candidato[v2].insert(j,i)
                        costo_cambio = costo(candidato,tipo)
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
                    contar += 1
                    #if contar == 1 :
                        #print("entre",actual,min_i,min_j,ciudad)
                     #   flag = False

            #if flag == False:
             #   break

        # Actualiza la subruta se encontró
        if actual < 0:
            ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]

def two_point_move(solucion):
    """
    Inserta un nodo en todas las posibles posiciones de otras rutas
    Se queda con el mejor vecino
    """
    if tipo == "minsum":
        v = random.randint(0,m-1)
    else:
        #v = max([(len(solucion[i]),i) for i in range(m)] , key=lambda x:x[0] )[1]
        v = max([(costo_parcial(solucion[i]),i) for i in range(m)],key=lambda x:x[0])[1]
    
    mejor_costo = costo(solucion,tipo)
    ruta_original = copy.deepcopy(solucion)
    candidato = copy.deepcopy(solucion)
    for i in solucion[v]:
        for v2 in range(m):
            if v2 != v:
                for j in range(len(solucion[v2])+1):
                    candidato[v] , candidato[v2] = candidato[v2], candidato[v]
                    costo_cambio = costo(candidato,tipo)
                    if costo_cambio < mejor_costo:
                        mejor_costo = costo_cambio
                        solucion = copy.deepcopy(candidato)
                        return 
                    candidato = copy.deepcopy(ruta_original)

def three_point_move(solucion):
    pass

def SA(seed,tipo):
    random.seed(seed)
    
    inicio = time.time()
    s = generar_solucion()
    #s = random_solution(n,m)
    costoActual = costo(s,tipo)

    s_mejor = copy.deepcopy(s)
    costoMejor = costoActual

    #Listas para graficar
    lista_costos = []
    lista_temperaturas = []

    iterMax = 3000
    temperatura = 100
    temperatura_final = 0.00001
    #alfa = (-math.log(temperatura)/math.log(temperatura_final))**(1/(iterMax-1))
    alfa = 0.9995
    #print(0,"%.2f"%costoMejor)
    # #graficar(s_b,tipo)

    max_vecinos = 5
    iteracion = 0
    #graficar(s_mejor,tipo)
    #while temperatura > temperatura_final and iteracion < iterMax:
    while time.time()-inicio < n:
        for i in range(max_vecinos):
            s_candidato = copy.deepcopy(s)
            aleatorio = random.uniform(0,1)
            perturbacion1(s_candidato)
            if aleatorio<2/3:
                two_point_move(s_candidato)
            else:
                two_opt_move(s_candidato)
            #bl = one_point_move if random.uniform(0,1)<0.5 else two_opt_move
            #bl(s_candidato)
            #two_opt_move(s_candidato)

            costoCandidato = costo(s_candidato,tipo)
            
            if costoCandidato < costoActual:
                costoActual, s = costoCandidato, copy.deepcopy(s_candidato)
                
                if costoCandidato < costoMejor:
                    costoMejor, s_mejor = costoCandidato, copy.deepcopy(s_candidato)
            
            elif random.uniform(0, 1) < math.exp(-abs(costoCandidato - costoActual) / temperatura):
                costoActual, s = costoCandidato, copy.deepcopy(s_candidato)
        #print(iteracion, "%.2f"%costoMejor)#,s_mejor)
        print(f"{ins} {m} {iteracion} {round(costoMejor,2)} {round(time.time()- inicio,2)}",end="\r")
        if iteracion in [3000,6000,9000]:
            temperatura = 100
        lista_costos.append(costoMejor)
        lista_temperaturas.append(temperatura)
        

        
        temperatura *= alfa
        iteracion +=1
    #print(ins,m,"%.2f"%costoMejor,"%.2f"%temperatura,"%.2f"%(time.time()-inicio))
    print()
    # try:
    #     #print("holac")
    #     graficar(s_mejor,tipo)
    #     #print(s_mejor)
    # except:
    #     pass
    #input_graficar = input("Graficar temperatura y costos? (Si/No): ").lower()
    input_graficar = "No"

    if input_graficar == "Si":
        graficar_temperatura_costos(lista_costos,lista_temperaturas)

instancias = ["11a","11b","12a","12b","16","51","100","128","150"]#["100","128","150"]
ms = {"11a":[3],"11b":[3],"12a":[3],"12b":[3],"16":[3],"128":[10,15,30],"51":[3,5,10,20],"100":[3,5,10,20],"150":[3,5,10,20]}
instancia = "11a"
tipos = ["minmax","minsum"][::-1]

#sol_inicial = [[1,2,3],[4,5,6],[7,8,9,10]]

for tipo in tipos:
    for ins in instancias:
        distancia,coords = read_data(ins)
        n = len(distancia)
        for ma in ms[ins]:
            m = ma
            for seed in range(1):
                SA(seed,tipo)
    print("")


# sol_inicial = [[80],
#                [49],
#                [26],
#                [23,87,78,12],
#                [i for i in range(1,100) if i not in [80,49,26,23,87,78,12] ]] 
#print(checker(sol_inicial))


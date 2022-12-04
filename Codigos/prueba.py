lista1 = [[1,2,3],[4,10,6],[7,9,8]]

lista2 = [[1,2,3,4,10,6,7,9,8],[0,0,1,0,0,1,0,0,1]]

aux = []
for i in range(len(lista2[0])):
    if lista2[1][i]==0:
        aux.append(lista2[0][i])
    else:
        aux.append(lista2[0][i])
        print(aux)
        aux = []
class Nodo:
    def __init__(self,dato):
        self.dato = dato
        self.h_izq = None
        self.h_der = None

class Arbol:
    def __init__(self):
      self.raiz = None
    def insertarNodo(dato,raiz):
        if raiz == None:
            raiz = Nodo(dato)
        else:
            if dato > raiz.dato:
                raiz.h_der = Arbol.insertarNodo(dato,raiz.h_der)
            else:
                raiz.h_izq = Arbol.insertarNodo(dato,raiz.h_izq)
                return raiz

    def insertar(self,dato):
        if self.raiz == None:
            self.raiz = Nodo(dato)  
        else:
            if dato > self.raiz.dato:
                self.raiz.h_der = Arbol.insertarNodo(dato,self.raiz.h_der)
            else:
                self.raiz.h_izq = Arbol.insertarNodo(dato,self.raiz.h_izq)
    def buscar(self,dato):
        return Arbol.buscarNodo(dato,self.raiz)

    def buscarNodo(dato,raiz):
        if raiz != None:
            if dato == raiz.dato:
                return True
             else:
                if dato > raiz.dato:
                    return Arbol.buscarNodo(dato,raiz.h_der)
                else:
                    return Arbol.buscarNodo(dato,raiz.h_izq)
        else:
            return False
    def sumar(raiz):
        if raiz==None:
            return 0
        else:
            if raiz.esHoja():
                return 1
            else :
                return 1 + Contar(raiz.h_izq)+ Contar(raiz.h der)
     
a = Arbol()
a.insertar(6)
a.insertar(2)
a.insertar(4)
a.insertar(9)
a.insertar(1)
a.insertar(5)
a.mostrar()
#Nos podria servir.
def contar(raiz):
    if raiz == None:
      return 0
    else:
      if raiz.h_izq==None and raiz.h_der==None:
        return 1
      else:
        return 1 + Arbol.contar(raiz.h_izq) + Arbol.contar(raiz.h_der)

  def altura(self):
    return Arbol.alturaRaiz(self.raiz)
  def alturaRaiz(raiz):
    if raiz == None:
      return -1
    else:
      if raiz.h_izq == None and raiz.h_der == None:
        return 0
      else:
        return 1 + Arbol.mayor(Arbol.alturaRaiz(raiz.h_izq),Arbol.alturaRaiz(raiz.h_der))
  def mayor(a,b):
    if a>b:
      return a
    else:
      return b
  def calcularPromedioHojas(self):
    pass


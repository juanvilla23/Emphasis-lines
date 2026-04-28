class Punto2D:
    def __init__(self, coorX, coorY):
        self.coorX = coorX
        self.coorY = coorY

    def distanciaOrigen(self):
        return (self.coorX**2 + self.coorY**2)**(1/2)

    def distanciaPuntos(self, p2):
        return ((self.coorX - p2.coorX)**2 + (self.coorY - p2.coorY)**2)**(1/2)
    
p=Punto2D(10, 20)
q=Punto2D(30, 40)
print(p.distanciaOrigen())
print(p.distanciaPuntos(q))
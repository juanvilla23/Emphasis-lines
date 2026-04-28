import numpy as np
data = [0,2,4,6]
res2=np.array(data)
print(f"Creación de arreglo con Numpy: {res2}")

res3=res2.shape
print(f"Filas y Columnas del arreglo: {res3}")

res4=res2.ndim
print(f"Filas del Arreglo: {res4}")

res5=res2.size
print(f"Tamaño del arreglo: {res5}")

data = [[0, 2, 4, 6], [1, 'a', 5, 7]]
res8=np.array(data)
print(f"Arreglo de datos: {res8}")

data = [(0, 2, 4, 6), (1, 'a', 5, 7)]
res9=np.array(data)
print(f"Arreglo de datos: {res9}")

data = [[0, 2, 4, 6], (1, 'a', 5, 7, 9)]
res10=np.array(data, dtype=object)
print(f"Arreglo de datos: {res10}")

res11=res10[1][4]
print(f"Elemento en la posición [1][4]: {res11}")

res12=np.arange(2, 10, 3)
print(f"Arreglo de datos: {res12}")

res14=np.zeros((3, 3))
print(f"Arreglo de ceros: {res14}")
print(res14[1,2])
res14[1,1]=1
res14[0,1]=2
print(res14[1,:])
print(res14[:,1])

edades = np.array([23, 56, 67, 89, 23, 56, 27, 12, 8, 72])
res18 = edades > 30
print(f"Edades mayores a 30: {res18}")
generos = np.array(['m', 'm', 'f', 'f', 'm', 'f', 'm', 'm', 'm', 'f'])
res19 = generos == 'm'
print(f"Generos masculinos: {res19}")

res20 = (edades > 10) & (generos == 'f')
print(f"Edades mayores a 10 y genero femenino: {res20}")

positivos = generos == 'm'
res21 = edades[positivos]
print(f"Edades de los hombres: {res21}")

res31=np.bincount(edades)
print(f"Frecuencia de edades: {res31}")





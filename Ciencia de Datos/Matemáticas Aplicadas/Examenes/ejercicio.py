

x = int(input("Ingrese un numero: "))

def total_numeros(x):
    t = 0
    while x > 10:
        t += 1
        x = x // 10
    return t

def quitarnumeros(x, n):
    return x // (10**n)


while x > 10:
    cantidad_numeros_iter = total_numeros(x)
    x_final = x % 10
    x_1 = quitarnumeros(x, cantidad_numeros_iter)
    if x_final != x_1:
        print("No es capicua")
        exit(0)
    x = x - x_1 * 10**(cantidad_numeros_iter)
    x = x % 10
print("Es capicua")

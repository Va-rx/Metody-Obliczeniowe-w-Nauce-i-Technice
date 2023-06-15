import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


def f(x):
    return np.exp(-3*np.sin(x)) + 3*np.sin(x) - 1


def lagrange(x, y, x_values):
    n = len(x)
    p = np.zeros_like(x_values)
    for k in range(n):
        L = 1.0
        for i in range(n):
            if i != k:
                L *= (x_values - x[i]) / (x[k] - x[i])
        p += y[k] * L
    return p


def newton(x, y, x_values):
    n = len(x)
    arr = [[0 for _ in range(n)] for _ in range(n)]

    for k in range(n):
        arr[0][k] = y[k]

    for k in range(1, n):
        arr[1][k] = (arr[0][k] - arr[0][k-1]) / (x[k] - x[k-1])

    for j in range(2, n):
        c = 0
        for k in range(j, n):
            arr[j][k] = (arr[j-1][k] - arr[j-1][k-1]) / (x[k] - x[c])
            c += 1

    p = np.zeros_like(x_values)
    for i in range(len(x_values)):
        p[i] = arr[0][0]
        iloczyn = 1
        for j in range(1, n):
            iloczyn *= (x_values[i] - x[j-1])
            p[i] += arr[j][j]*iloczyn
    return p


def chebyshev_nodes(a, b, n):
    return np.array([((a+b)/2.0)+((b-a)/2.0) * np.cos(np.pi*(2.0*i - 1.0)/(2.0*n)) for i in range(1, n+1)])


def max_error(g):
    x1 = np.linspace(a, b, 1000)
    y1 = f(x1)

    diff = abs(y1-g)
    return max(diff)


def avg_error(g):
    x1 = np.linspace(a, b, 1000)
    y1 = f(x1)

    diff = mean(abs(y1-g))
    return diff


a, b = -3*np.pi, 4*np.pi  # Przedział
n = 5 # Liczba węzłów

# Sposób generacji węzłów
x = np.linspace(a, b, n)
# x = chebyshev_nodes(a, b, n)

# Kropki na wykresie jako punkty węzłów
y = f(x)
plt.plot(x, y, 'bo', label="f(x)")

# Wykres funkcji pierwotnej
xDraw = np.linspace(a, b, 1000)
y = f(xDraw)
plt.plot(xDraw, y)

# Punkty wyznaczone do narysowanie interpolacji
y = f(x)
x_values = np.linspace(a, b, 1000)
y_lagrange = lagrange(x, y, x_values)
# y_newton = newton(x, y, x_values)

# plt.plot(x_values, y_newton, label="Newton")
plt.plot(x_values, y_lagrange, label="Lagrange")

print("Najwieksza roznica: ", max_error(y_lagrange))  # Bład maksymalny dla podanej funkcji interpolacującej
print("Srednia roznica: ", avg_error(y_lagrange))  # Średni błąd dla -||-
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


def f(x):
    return np.exp(-3*np.sin(x)) + 3*np.sin(x) - 1


def df(x):
    return -3*np.cos(x)*np.exp(-3*np.sin(x)) + 3*np.cos(x)


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


def hermite(x, y, dy, x_values):
    n = 2*len(x)
    m = len(x)
    arr = [[0 for _ in range(n)] for _ in range(n)]

    # 1st column
    for i in range(m):
        arr[2*i][0] = y[i]
        arr[2*i+1][0] = y[i]

    # 1/2 2nd column
    for i in range(m):
        arr[2*i+1][1] = dy[i]

    # 1/2 2nd column
    for i in range(2, n, 2):
        arr[i][1] = (arr[i][0] - arr[i-1][0]) / (x[i//2] - x[(i-1)//2])

    for j in range(2, n):
        c = 0
        for k in range(j, n):
            arr[k][j] = (arr[k][j-1] - arr[k-1][j-1]) / (x[k//2] - x[c//2])
            c += 1

    p = np.zeros_like(x_values)
    for i in range(len(x_values)):

        p[i] = arr[0][0]
        iloczyn = 1
        for j in range(1, n):
            iloczyn *= (x_values[i] - x[(j-1)//2])
            p[i] += arr[j][j]*iloczyn

    return p


a, b = -3*np.pi, 4*np.pi  # Przedział
n = 10 # Liczba węzłów

# Sposób generacji węzłów
# x = np.linspace(a, b, n)
x = chebyshev_nodes(a, b, n)

# Kropki na wykresie jako punkty węzłów
y = f(x)
plt.plot(x, y, 'bo', label="f(x)")

# Wykres funkcji pierwotnej
xDraw = np.linspace(a, b, 1000)
y = f(xDraw)
plt.plot(xDraw, y)

# Punkty wyznaczone do narysowanie interpolacji
y = f(x)
dy = f(x)
x_values = np.linspace(a, b, 1000)
y_hermit = hermite(x, y, dy, x_values)

plt.plot(x_values, y_hermit, label="hermit")

print("Najwieksza roznica: ", max_error(y_hermit))  # Bład maksymalny dla podanej funkcji interpolacującej
print("Srednia roznica: ", avg_error(y_hermit))  # Średni błąd dla -||-
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


def f(x):
    return np.exp(-3 * np.sin(x)) + 3 * np.sin(x) - 1


def spline3(x, y, x_values, type):
    n = len(x)
    A = [[0 for _ in range(n)] for _ in range(n)]
    h = x[1] - x[0]

    # Tablica przechowujaca wartosci delty_i
    big_delta_array = [0 for _ in range(n - 1)]
    for i in range(0, n - 1):
        big_delta_array[i] = (y[i + 1] - y[i]) / h
    # big_delta_array: [0] = delta_0, [1] = delta_1, [n-2] = delta_{n-2}

    # Prawa strona równania
    B = [0 for _ in range(n)]
    for i in range(1, n - 1):
        B[i] = big_delta_array[i] - big_delta_array[i - 1]

    if type == "natural":
        # Lewa strona równania
        A[0][0] = 1
        A[n - 1][n - 1] = 1
        for i in range(1, n - 1):
            A[i][i - 1] = h
            A[i][i] = 4 * h
            A[i][i + 1] = h
    elif type == "cubic":
        B[0] = h * h * big_delta_array[0]
        B[n - 1] = -1 * h * h * big_delta_array[n - 3]

        # Lewa strona równania
        A[0][0] = -1 * h
        A[0][1] = h
        A[n - 1][n - 1] = -1 * h
        A[n - 1][n - 2] = h
        for i in range(1, n - 1):
            A[i][i - 1] = h
            A[i][i] = 4 * h
            A[i][i + 1] = h

    sigma_array = np.linalg.solve(A, B)

    # Wyliczanie wspolczynnikow
    b_array = [0 for _ in range(n - 1)]
    c_array = [0 for _ in range(n - 1)]
    d_array = [0 for _ in range(n - 1)]
    for i in range(n - 1):
        b_array[i] = (y[i + 1] - y[i]) / h - h * (sigma_array[i + 1] + 2 * sigma_array[i])
        c_array[i] = 3 * sigma_array[i]
        d_array[i] = (sigma_array[i + 1] - sigma_array[i]) / h

    # Podkladanie pod S_i(x)
    result = [0 for _ in range(len(x_values))]
    current = 0
    for i in range(len(x_values)):
        if x_values[i] > x[current + 1]:
            current += 1
        result[i] = y[current] + b_array[current] * (x_values[i] - x[current]) + \
                    c_array[current] * ((x_values[i] - x[current]) ** 2) + \
                    d_array[current] * ((x_values[i] - x[current]) ** 3)

    return result


def spline2(x, y, x_values):
    n = len(x)
    A = [[0 for _ in range(3*(n-1))] for _ in range(3*(n-1))]

    # Prawa strona równania

    # Wszystkie wartości są równe 0 oprócz tych równan gdzie podkładamy współrzędne punktu
    # Do S_i(x_i)
    B = [0 for _ in range(3*(n-1))]

    # Warunek dla x_1 oraz x_n
    B[0] = y[0]
    B[2*(n-1)-1] = y[n-1]

    count = 1
    # Pozostałe wartości różne od 0
    for i in range(1, 2*(n-1)-2, 2):
        B[i] = y[count]
        B[i+1] = y[count]
        count += 1

    # Koniec prawej strony równania

    # Lewa strona równania

    counter = 0
    x_index = 0
    for i in range(0, 2*(n-1)-1, 2):
        A[i][counter + 0] = y[x_index]*y[x_index]
        A[i][counter + 1] = y[x_index]
        A[i][counter + 2] = 1

        A[i+1][counter + 0] = y[x_index+1]*y[x_index+1]
        A[i+1][counter + 1] = y[x_index+1]
        A[i+1][counter + 2] = 1

        counter += 3
        x_index += 1

    #--
    counter = 0
    x_index = 1
    for i in range(2*(n-1), 3*(n-1)-1):
        A[i][counter + 0] = 2*y[x_index]
        A[i][counter + 1] = 1
        A[i][counter + 3] = -2*y[x_index]
        A[i][counter + 4] = -1

        counter += 3
        x_index += 1

    A[3*(n-1)-1][0] = 2*y[0]
    A[3*(n-1)-1][1] = 1

    # array[0] = a_1, array[1] = b_1, array[2] = c_1, array[3] = a_2 ......
    array = np.linalg.solve(A, B)

    # Podkladanie pod S_i(x)
    result = [0 for _ in range(len(x_values))]
    current = 0
    iterator = 0
    for i in range(len(x_values)):
        if x_values[i] > x[current + 1]:
            current += 1
            iterator += 3
        result[i] = array[iterator]*(x_values[i])*(x_values[i]) + \
            array[iterator+1]*(x_values[i]) + array[iterator+2]
    return result


def spline2v2(x, y, x_values):
    n = len(x)
    A = [[0 for _ in range(n)] for _ in range(n)]
    B = [0 for _ in range(n)]
    h = x[1] - x[0]

    A[0][0] = 1
    A[n-1][n-1] = 1
    for i in range(1, n-1):
        A[i][i] = 1
        A[i][i+1] = 1

    for i in range(1, n-1):
        B[i] = 2*(y[i+1] - y[i]) / h

    b_array = np.linalg.solve(A, B)

    c_array = [0 for _ in range(n-1)]
    for i in range(n-1):
        c_array[i] = (b_array[i+1] - b_array[i]) / 2*h

    # Podkladanie pod S_i(x)
    result = [0 for _ in range(len(x_values))]
    current = 0
    for i in range(len(x_values)):
        if x_values[i] > x[current + 1]:
            current += 1
        result[i] = y[current] + b_array[current]*(x_values[i] - x[current]) + c_array[current]* \
                    ((x_values[i] - x[current])**2)
    return result


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


a, b = -3 * np.pi, 4 * np.pi  # Przedział
n = 10 # Liczba węzłów

# Sposób generacji węzłów
x = np.linspace(a, b, n)

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

y_new_spline = spline2v2(x, y, x_values)
y_new_spline2 = spline2(x, y, x_values)

y_spline_natural = spline3(x, y, x_values, "natural")
y_spline_cubic = spline3(x, y, x_values, "cubic")

plt.plot(x_values, y_new_spline2, label="spline2")
# plt.plot(x_values, y_new_spline, label="new")
# plt.plot(x_values, y_spline_natural, label="Natural")
# plt.plot(x_values, y_spline_cubic, label="Cubic")

# print("Najwieksza roznica: ", max_error(y_spline_natural))  # Bład maksymalny dla podanej funkcji interpolacującej
# print("Srednia roznica: ", avg_error(y_spline_natural))  # Średni błąd dla -||-
#
# print("Najwieksza roznica: ", max_error(y_spline_cubic))  # Bład maksymalny dla podanej funkcji interpolacującej
# print("Srednia roznica: ", avg_error(y_spline_cubic))  # Średni błąd dla -||-

plt.legend()
plt.show()

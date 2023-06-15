from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def f(x):
    return np.exp(-3*np.sin(x)) + 3*np.sin(x) - 1


def aprox(x, y, m, x_values):
    n = len(x)
    B = [0 for _ in range(m)]
    A = [[0 for _ in range(m)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            B[i] += (y[j]) * (x[j])**i

    for i in range(m):
        for j in range(m):
            for k in range(n):
                A[i][j] += x[k]**(i+j)

    solved = np.linalg.solve(A, B)

    result = [0 for _ in range(len(x_values))]
    for i in range(len(x_values)):
        for j in range(m):
            result[i] += solved[j] * x_values[i] ** j
    return result


# n > (2m+1) [liczba wyliczanych wspolczynnikow]
def aproxTrig(x, y, m, x_values):
    n = len(x)

    A = [0 for _ in range(m)]
    B = [0 for _ in range(m)]

    for i in range(m):
        summ = 0
        for j in range(n):
            summ += y[j]*np.cos(i*x[j])
        A[i] = 2/n * summ

    for i in range(m):
        summ = 0
        for j in range(n):
            summ += y[j]*np.sin(i*x[j])
        B[i] = 2/n * summ

    result = [0 for _ in range(len(x_values))]
    for i in range(len(x_values)):
        for j in range(m):
            summ = 0
            for k in range(1, j):
                summ += A[k] * np.cos(k * x_values[i]) + B[k] * np.sin(k * x_values[i])
            result[i] += 1/2 * A[0] + summ
    return result


# Draw a basic function, without .show()
def draw_basic():
    # Points
    y = f(x)
    plt.plot(x, y, 'bo', label="F(x)")

    # Function F
    y = f(xDraw)
    plt.plot(xDraw, y)


# Draw a function where n is const for few m's in loop
def draw_few_m_algebraic(m_start, m_end):
    for em in range(m_start, m_end+1):
        draw_basic()

        # Function f
        y = f(x)
        aproximated = aprox(x, y, em, xDraw)
        plt.plot(xDraw, aproximated, label="f(x)")
        plt.legend()

        path = "Graphs/Alg/n" + str(n)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        print("m:", em, ", error:", max_error(aproximated), ", avg:", avg_error(aproximated))

        file = path + "/n" + str(n) + "m" + str(em) + "alg.png"
        print(file)
        plt.savefig(file, bbox_inches='tight')

        plt.show()


# Draw a function for specific n and m
def draw_one_algebraic():
    draw_basic()

    #Function f
    y = f(x)
    aproximated = aprox(x, y, m, xDraw)
    plt.plot(xDraw, aproximated, label="f(x)")
    plt.legend()

    file = "Graphs/Alg/n" + str(n) + "m" + str(m) + "alg.png"
    print(file)
    plt.savefig(file, bbox_inches='tight')

    plt.show()


# Max error between main function and approximated
def max_error(g):
    x1 = np.linspace(a, b, 1000)
    y1 = f(x1)

    diff = abs(y1-g)
    return max(diff)


# Average error between main function and approximated
def avg_error(g):
    x1 = np.linspace(a, b, 1000)
    y1 = f(x1)

    diff = mean(abs(y1-g))
    return diff


a, b = -3*np.pi, 4*np.pi  # Przedział
n = 30  # Liczba węzłów
m = 4  # Liczba funkcji bazowych (nie stopień wielomianu)
x = np.linspace(a, b, n)  # węzły x_0, ... x_n-1
xDraw = np.linspace(a, b, 1000)  # Punkty na podstawie których rysujemy

# draw_one_algebraic()
draw_few_m_algebraic(3, n)

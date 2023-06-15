from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def f(x):
    return np.exp(-3*np.sin(x)) + 3*np.sin(x) - 1


def aproxTrig(x, y, m, x_values):
    n = len(x)
    if m - 1 > np.floor((n-1)/2):
        raise ValueError

    a = x[0]
    b = x[-1]
    a_trans = -np.pi
    b_trans = np.pi

    def transform_x(x, a, b, a_trans, b_trans):
        return ((x - a) / (b - a)) * (b_trans - a_trans) + a_trans

    def calc_ak(x, y, k):
        n = len(x)
        return (2 / n) * np.sum(y * np.cos(k * x))

    def calc_bk(x, y, k):
        n = len(x)
        return (2 / n) * np.sum(y * np.sin(k * x))

    xs = transform_x(x, a, b, a_trans, b_trans)
    ak = np.array([calc_ak(xs, y, k) for k in range(m)])
    bk = np.array([calc_bk(xs, y, k) for k in range(m)])

    def f(x):
        x_trans = transform_x(x, a, b, a_trans, b_trans)
        return 0.5 * ak[0] + np.sum(ak[1:] * np.cos(np.arange(1, m) * x_trans) + bk[1:] * np.sin(np.arange(1, m) * x_trans))

    result = [f(x) for x in x_values]
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
def draw_few_m_trig(m_start, m_end):
    for em in range(m_start, m_end+1):
        draw_basic()

        # Function f
        y = f(x)
        aproximated = aproxTrig(x, y, em, xDraw)
        plt.plot(xDraw, aproximated, label="f(x)")
        plt.legend()

        path = "Graphs/Trig/n" + str(n)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        print("m:", em, ", error:", max_error(aproximated), ", avg:", avg_error(aproximated))

        file = path + "/n" + str(n) + "m" + str(em) + "trig.png"
        print(file)
        plt.savefig(file, bbox_inches='tight')

        plt.show()


# Draw a function for specific n and m
def draw_one_trig():
    draw_basic()

    #Function f
    y = f(x)
    aproximated = aproxTrig(x, y, m, xDraw)
    plt.plot(xDraw, aproximated, label="f(x)")
    plt.legend()

    file = "Graphs/Trig/n" + str(n) + "m" + str(m) + "trig.png"
    print(file)
    plt.savefig(file, bbox_inches='tight')
    print("Max: ", max_error(aproximated))
    print("Avg: ", avg_error(aproximated))

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
n = 9  # Liczba węzłów
m = 4  # Liczba funkcji bazowych (nie stopień wielomianu)
x = np.linspace(a, b, n)  # węzły x_0, ... x_n-1
xDraw = np.linspace(a, b, 1000)  # Punkty na podstawie których rysujemy

# draw_basic()
# draw_one_trig()
draw_few_m_trig(m, n)

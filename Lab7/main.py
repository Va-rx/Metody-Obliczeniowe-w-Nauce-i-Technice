import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (x-1) * np.exp(-13*x) + x**15


def fp(x):
    return np.exp(-13*x) * (15*x**14*np.exp(13*x) - 13*x + 14)


def newton(rho2, rho_type):
    x = -0.1
    i = 0
    if rho_type == 1:
        while np.abs(f(x)) > rho2:
            x = x - (f(x) / fp(x))
            i = i + 1
    elif rho_type == 2:
        while True:
            prev = x
            x = x - (f(x) / fp(x))
            i = i + 1
            if np.abs(x - prev) <= rho2:
                break
    print("Iteracji: ", i)
    print("Wartosc: ", x)


def sieczne(rho2, rho_type):
    x1 = 1.8
    x2 = 1.9
    x3 = None
    i = 0
    if rho_type == 1:
        while True:
            x3 = x2 - (x2 - x1) / (f(x2) - f(x1)) * f(x2)
            i = i + 1
            if np.abs(np.abs(f(x3)) <= rho2):
                break
            x1 = x2
            x2 = x3
    elif rho_type == 2:
        while True:
            prev = x3
            x3 = x2 - (x2 - x1) / (f(x2) - f(x1)) * f(x2)
            i = i + 1
            if prev is not None and np.abs(np.abs(x3 - prev) <= rho2):
                break
            x1 = x2
            x2 = x3
    print("Iteracji: ", i)
    print("Wartosc: ", x3)


rho_type = 2 # 1 dla warunku f(x_i), 2 dla warunku x_i - x_{i-1}
a = -0.1
b = 1.9
ro = 10**(-2)
# newton(10**(-2), rho_type)
# newton(10**(-5), rho_type)
# newton(10**(-8), rho_type)
# newton(10**(-10), rho_type)

sieczne(10**(-2), rho_type)
sieczne(10**(-5), rho_type)
sieczne(10**(-8), rho_type)
sieczne(10**(-10), rho_type)

xDraw = np.linspace(a, b, 1000)
y = f(xDraw)
# plt.plot(xDraw, y, label="f(x)")
# plt.legend()
# plt.show()



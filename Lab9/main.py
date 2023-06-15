import numpy as np
import matplotlib.pyplot as plt
import time


def spectral_radius(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    M = np.linalg.inv(D).dot(L + U)

    eigenvalues = np.linalg.eigvals(M)

    radius = max(abs(eigenvalues))

    return radius


def create_arrays(n, dtype=np.float64):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        A[i][i] = 8
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i][j] = 1 / (np.abs(i - j + 2) + 3)

    X = np.ones(n, dtype=dtype)
    for i in range(0, n, 3):
        X[i] *= -1

    B = np.zeros(n, dtype=dtype)
    for i in range(n):
        val = 0
        for j in range(n):
            val += A[i][j] * X[j]
        B[i] = val
    return A, X, B


def find_max_error(true_values, result_values):
    return np.max(np.abs(true_values - result_values))


def find_errors(true_values, result_values):
    return np.abs(true_values - result_values)


def draw(errors):
    indexes = range(len(errors))

    plt.plot(indexes, errors, 'bo')
    plt.xticks(indexes, [str(i) for i in indexes])
    plt.show()


def draw_graph(iterat1, iterat2, n):
    plt.plot(n, iterat1, 'bo', color='red', label='Wektor zerowy')
    plt.plot(n, iterat2, 'bo', label='Wektor odległy')
    plt.xticks(n, [str(i) for i in n])
    plt.title("Wykres przedstawiający normę maksimum dla q=1e-10, kryterium 1.")
    plt.xlabel("Wielkość macierzy (n)")
    plt.ylabel("Norma maksimum")
    plt.legend()
    plt.show()


def jacobi_method(A, b, rho_type, rho=1e-10, max_iterations=1000):
    n = len(A)
    x = np.zeros_like(b, dtype=np.double)
    # x = np.ones(n)
    # for i in range(0, n, 3):
    #     x[i] *= -100

    D = np.diag(A)
    R = A - np.diagflat(D)

    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D

        # kryterium stopu: || x_{i+1} - x_i || < rho
        if rho_type == 1 and np.linalg.norm(x_new - x) < rho:
            break
        if rho_type == 2 and np.linalg.norm(np.dot(A, x) - b) < rho:
            break

        x = x_new

    return x, i


def main():
    N_array = [5, 10, 15, 20, 30, 50, 75, 100, 120, 150, 200]

    n_max = max(N_array)
    X_result1Array = np.zeros((len(N_array), n_max))
    X_result2Array = np.zeros((len(N_array), n_max))
    Iterations1Array = np.zeros(len(N_array))
    Iterations2Array = np.zeros(len(N_array))
    Max_errors1 = np.zeros(len(N_array))
    Max_errors2 = np.zeros(len(N_array))
    Spectral_radius = np.zeros(len(N_array))
    Time1 = np.zeros(len(N_array))
    Time2 = np.zeros(len(N_array))

    for i, n in enumerate(N_array):
        A_array, X_array, B_array = create_arrays(n)
        start_time1 = time.time()
        X_result1Array[i, :n], Iterations1Array[i] = jacobi_method(A_array, B_array, 1, 1e-10, max_iterations=1000)
        end_time1 = time.time()
        start_time2 = time.time()
        X_result2Array[i, :n], Iterations2Array[i] = jacobi_method(A_array, B_array, 2, 1e-15, max_iterations=1000)
        end_time2 = time.time()
        Max_errors1[i] = find_max_error(X_array, X_result1Array[i, :n])
        Max_errors2[i] = find_max_error(X_array, X_result2Array[i, :n])
        Time1[i] = end_time1 - start_time1
        Time2[i] = end_time2 - start_time2
        Spectral_radius[i] = spectral_radius(A_array)

    print("Max error\n", Max_errors1)
    # print("Promien spektralny\n", Spectral_radius)
    # print("Liczba iteracji\n", Iterations1Array)
    # print(Time2)
    # print(Time2)
    # draw_graph(Time1, Time2, N_array)
    # draw_graph(Max_errors1, Max_errors2, N_array)


if __name__ == "__main__":
    main()

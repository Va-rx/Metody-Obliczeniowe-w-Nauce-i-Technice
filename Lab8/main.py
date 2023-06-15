import numpy as np
import matplotlib.pyplot as plt
import time


def conditioning(A):
    inv_A = np.linalg.inv(A)

    norm_A = np.linalg.norm(A)
    norm_inv_A = np.linalg.norm(inv_A)

    return norm_A * norm_inv_A


def create_arrays(n, dtype):
    A = np.zeros((n, n), dtype=dtype)
    for j in range(n):
        A[0][j] = 1
    for i in range(1, n):
        for j in range(n):
            A[i][j] = 1 / (i + j + 1)

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


def create_arrays2(n, dtype):
    A = np.zeros((n, n), dtype=dtype)
    for j in range(n):
        A[0][j] = 1
    for i in range(1, n):
        for j in range(n):
            A[i][j] = 1 / (i + j + 1)

    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i][j] = 2*(i+1)/(j+1)
            else:
                A[i][j] = A[j][i]

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


def create_arrays3(n, dtype):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        A[i][i] = -2*(i+1)-4
    for i in range(n-1):
        A[i][i+1] = i + 1
    for i in range(1, n):
        A[i][i-1] = 2/i
    for i in range(n):
        for j in range(n):
            if i > j + 1 > i + 2:
                A[i][j] = 0

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


def solve_x(A_copy, B_copy, dtype):
    A, B = np.copy(A_copy), np.copy(B_copy)
    A = A.astype(dtype)
    B = B.astype(dtype)
    n = len(A)

    for i in range(n):
        for j in range(i + 1, n):
            multiplier = -A[j][i] / A[i][i]
            for k in range(n):
                A[j][k] += multiplier * A[i][k]
            B[j] += multiplier * B[i]

    X = np.zeros(n)
    X[n - 1] = B[n - 1] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        sum_val = 0
        for j in range(i + 1, n):
            sum_val += A[i][j] * X[j]
        X[i] = (B[i] - sum_val) / A[i][i]
    return X


def solve_x_thomas(A_copy, B_copy, dtype):
    n = np.shape(A_copy)[0]
    C = np.zeros(n)
    A = np.copy(A_copy)
    B = np.copy(B_copy)
    A = A.astype(dtype)
    B = B.astype(dtype)
    C[0] = A[0][0]

    X = np.zeros(n)
    X[0] = B[0]

    for i in range(1, n):
        ratio = A[i][i - 1] / C[i - 1]
        C[i] = A[i][i] - ratio * A[i - 1][i]
        X[i] = B[i] - ratio * X[i - 1]

    X[n - 1] = X[n - 1] / C[n - 1]
    for i in range(n - 2, -1, -1):
        X[i] = (X[i] - A[i][i + 1] * X[i + 1]) / C[i]
    return X


def find_max_error(true_values, result_values):
    return np.max(np.abs(true_values - result_values))


def find_errors(true_values, result_values):
    return np.abs(true_values - result_values)


def draw(errors):
    indexes = range(len(errors))

    plt.plot(indexes, errors, 'bo')
    plt.xticks(indexes, [str(i + 3) for i in indexes])
    plt.show()


def draw6432(errors64, errors32):
    indexes = range(len(errors64))

    plt.plot(indexes, errors64, 'bo', label='float64')
    plt.plot(indexes, errors32, 'bo', label='float32', color="red")
    plt.xticks(indexes, [str(i + 3) for i in indexes])
    plt.legend()
    plt.show()


def main():
    n = 12
    dtype = np.float64
    # A, X_known, B = create_arrays(n, dtype)
    # X_result = solve_x(A, B, dtype)
    # avg_error = find_avg_error(X_known, X_result)
    # errors = find_errors(X_known, X_result)

    Max_errors64 = np.zeros(18, dtype=dtype)
    cond = np.zeros(18)
    for i in range(3, 21):
        Ai, Xi_known, Bi = create_arrays3(i, dtype)
        start_time = time.time()
        Xi_result = solve_x(Ai, Bi, dtype)
        end_time = time.time()
        Max_errors64[i - 3] = find_max_error(Xi_known, Xi_result)
        cond[i-3] = conditioning(Ai)
        print("n=", i, ": ", end_time - start_time, "s")

    dtype = np.float32

    Max_errors32 = np.zeros(18, dtype=dtype)
    for i in range(3, 21):
        Ai, Xi_known, Bi = create_arrays3(i, dtype)
        start_time = time.time()
        Xi_result = solve_x(Ai, Bi, dtype)
        end_time = time.time()
        Max_errors32[i - 3] = find_max_error(Xi_known, Xi_result)
        print("n=", i, ": ", end_time - start_time, "s")

    print("Dla 64: ", Max_errors64)
    print("Dla 32: ", Max_errors32)

    draw6432(Max_errors64, Max_errors32)


if __name__ == "__main__":
    main()

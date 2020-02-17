import numpy as np


def choleski_method(a, b):
    n = a.shape[0]
    u = np.zeros([n, n])
    u_inverse = np.zeros([n, n])

    def iterate_1(i):
        sigma = 0
        for k in range(i):
            sigma = sigma + u[k, i] ** 2
        u_ii = (a[i, i] - sigma) ** (1/2)
        return u_ii

    def iterate_2(i, j):
        sigma = 0
        for k in range(i):
            sigma = sigma + u[k, i] * u[k, j]
        u_ij = (1 / u[i, i]) * (a[i, j] - sigma)
        return u_ij

    def iterate_3(i, j):
        sigma = 0
        for k in range(i + 1, j + 1):
            sigma = sigma + u[i, k] * u_inverse[k, j]
        ui_ij = -sigma / u[i, i]
        return ui_ij

    u[0, 0] = a[0, 0] ** (1 / 2)
    for j in range(1, n):
        u[0, j] = a[0, j] / u[0, 0]
    for i in range(1, n):
        for j in range(i, n):
            if i == j:
                u[i, i] = iterate_1(i)
            else:
                u[i, j] = iterate_2(i, j)
    for i in range(n):
        u_inverse[i, i] = 1 / u[i, i]
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i < j:
                u_inverse[i, j] = iterate_3(i, j)
            elif i > j:
                u_inverse[i, j] = 0
            else:
                continue
    a_inverse = np.dot(u_inverse, u_inverse.T)
    x = np.dot(a_inverse, b)
    return x


def gauss_elimination(a, b):
    c = np.hstack((a, b))
    c_old = c.copy()
    n = c.shape[0]

    def summation(c, x):
        s = 0
        for j in range(i + 1, n):
            s = s + c[i, j] * x[j]
        return s

    for k in range(n - 1):
        for i in range(k + 1, n):
            for j in range(k, n + 1):
                c[i, j] = c_old[i, j] - (c_old[i, k] / c_old[k, k]) * c_old[k, j]
        c_old = c.copy()
    x = np.zeros(n)
    x[n - 1] = c_old[n - 1, n] / c_old[n - 1, n - 1]
    for i in range(n - 1, -1, -1):
        s = summation(c, x)
        x[i] = (c[i, n] - s) / c[i, i]
    return x


def gauss_jordan_elimination(a, b):
    c = np.hstack([a, b])
    c_old = c.copy()
    n = c.shape[0]
    for k in range(n):
        for j in range(k, n + 1):
            c[k, j] = c_old[k, j] / c_old[k, k]
        for j in range(k, n + 1):
            for i in range(n):
                if i == k:
                    c[i, j] = c[i, j]
                else:
                    c[i, j] = c_old[i, j] - c_old[i, k] * c[k, j]
        c_old = c.copy()
    x = c[:, n]
    return x


def gauss_seidel_iteration_method(a, b):
    global error, err_r, err_a, x
    n = a.shape[1]
    error_message = "Error. Enter a number for your selection."
    case = input("Input iterations (1) or convergence condition (2)? ")
    if case == "1":
        iterations = int(input("Enter number of iterations: "))
        converge = "N/A"
        e = "N/A"
    elif case == "2":
        converge = input("Input relative error (1) or absolute error (2)? ")
        e = float(input("Enter convergence condition as a decimal: "))
        iterations = "N/A"
    else:
        print(error_message)

    initial = input("Input an initial guess (1) or start with zero vector (2)? ")
    if initial == "1":
        guess = input("Enter initial guess, comma separated in brackets ([x1,x2,x3,...,xn]): ")
        x = np.fromstring(guess[1:-1], sep=',')
        x = x[np.newaxis, :]
        if x.size == n:
            print("Input accepted")
        elif x.size < n:
            print("Error. Not enough guess inputs.")
        elif x.size > n:
            print("Error. Too many guess inputs.")
        else:
            print("Error. Try again - format guess comma separated in brackets ([x1,x2,x3,...,xn])")
    elif initial == "2":
        x = np.zeros([1, n])
    else:
        print("Error. Enter 'y' or 'n'.")

    def iterate(i, k):
        sigma = 0
        for j in range(n):
            if j < i:
                sigma = sigma + a[i, j] * x_k1[j]
            elif j > i:
                sigma = sigma + a[i, j] * x[k, j]
            else:
                sigma = sigma
        x_ijk = 1 / a[i, i] * (b[i] - sigma)
        return x_ijk

    if case == "1":
        error = np.array([[0, 0]])
        for k in range(iterations):
            x_k1 = np.zeros(n)
            for i in range(n):
                x_k1[i] = iterate(i, k)
            x = np.vstack([x, x_k1])
            if x[k, :].all() != 0:
                xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
            else:
                xe_r = 1_000_000
            xe_r = np.abs(xe_r)
            err_r = np.max(xe_r) * 100
            xe_a = x[k + 1, :] - x[k, :]
            xe_a = np.abs(xe_a)
            err_a = np.max(xe_a)
            error = np.vstack([error, [err_a, err_r]])
    elif case == "2":
        err_r = 1_000_000
        err_a = 1_000_000
        error = np.array([[0, 0]])
        if converge == "2":
            k = 0
            while abs(err_a) > e:
                x_k1 = np.zeros(n)
                for i in range(n):
                    x_k1[i] = iterate(i, k)
                x = np.vstack([x, x_k1])
                xe_a = x[k + 1, :] - x[k, :]
                xe_a = np.abs(xe_a)
                err_a = np.max(xe_a)
                if x[k, :].all() != 0:
                    xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
                else:
                    xe_r = 1_000_000
                xe_r = np.abs(xe_r)
                err_r = np.max(xe_r)
                error = np.vstack([error, [err_a, err_r]])
                k += 1
                iterations = k
        elif converge == "1":
            k = 0
            while abs(err_r) > e:
                x_k1 = np.zeros(n)
                for i in range(n):
                    x_k1[i] = iterate(i, k)
                x = np.vstack([x, x_k1])
                if x[k, :].all() != 0:
                    xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
                else:
                    xe_r = 1_000_000
                xe_r = np.abs(xe_r)
                err_r = np.max(xe_r)
                xe_a = x[k + 1, :] - x[k, :]
                xe_a = np.abs(xe_a)
                err_a = np.max(xe_a)
                error = np.vstack([error, [err_a, err_r]])
                k += 1
                iterations = k
    else:
        print("Error. Enter a number for your selection.")
    return x, error, iterations


def jacobi_iteration_method(a, b):
    global error, err_r, err_a, x
    n = a.shape[1]
    error_message = "Error. Enter a number for your selection."
    case = input("Input iterations (1) or convergence condition (2)? ")
    if case == "1":
        iterations = int(input("Enter number of iterations: "))
        converge = "N/A"
        e = "N/A"
    elif case == "2":
        converge = input("Input relative error (1) or absolute error (2)? ")
        e = float(input("Enter convergence condition as a decimal: "))
        iterations = "N/A"
    else:
        print(error_message)

    initial = input("Input an initial guess (1) or start with zero vector (2)? ")
    if initial == "1":
        guess = input("Enter initial guess, comma separated in brackets ([x1,x2,x3,...,xn]): ")
        x = np.fromstring(guess[1:-1], sep=',')
        x = x[np.newaxis, :]
        if x.size == n:
            print("Input accepted")
        elif x.size < n:
            print("Error. Not enough guess inputs.")
        elif x.size > n:
            print("Error. Too many guess inputs.")
        else:
            print("Error. Try again - format guess comma separated in brackets ([x1,x2,x3,...,xn])")
    elif initial == "2":
        x = np.zeros([1, n])
    else:
        print("Error. Enter 'y' or 'n'.")

    def iterate(i, k):
        sigma = 0
        for j in range(n):
            if j != i:
                sigma = sigma + a[i, j] * x[k, j]
                j += 1
            else:
                j += 1
        x_ijk = 1 / a[i, i] * (b[i] - sigma)
        return x_ijk

    if case == "1":
        error = np.array([[0, 0]])
        for k in range(iterations):
            x_k1 = np.zeros(n)
            for i in range(n):
                x_k1[i] = iterate(i, k)
            x = np.vstack([x, x_k1])
            if x[k, :].all() != 0:
                xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
            else:
                xe_r = 1_000_000
            xe_r = np.abs(xe_r)
            err_r = np.max(xe_r) * 100
            xe_a = x[k + 1, :] - x[k, :]
            xe_a = np.abs(xe_a)
            err_a = np.max(xe_a)
            error = np.vstack([error, [err_a, err_r]])
    elif case == "2":
        err_r = 1_000_000
        err_a = 1_000_000
        error = np.array([[0, 0]])
        if converge == "2":
            k = 0
            while abs(err_a) > e:
                x_k1 = np.zeros(n)
                for i in range(n):
                    x_k1[i] = iterate(i, k)
                x = np.vstack([x, x_k1])
                xe_a = x[k + 1, :] - x[k, :]
                xe_a = np.abs(xe_a)
                err_a = np.max(xe_a)
                if x[k, :].all() != 0:
                    xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
                else:
                    xe_r = 1_000_000
                xe_r = np.abs(xe_r)
                err_r = np.max(xe_r)
                error = np.vstack([error, [err_a, err_r]])
                k += 1
                iterations = k
        elif converge == "1":
            k = 0
            while abs(err_r) > e:
                x_k1 = np.zeros(n)
                for i in range(n):
                    x_k1[i] = iterate(i, k)
                x = np.vstack([x, x_k1])
                if x[k, :].all() != 0:
                    xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
                else:
                    xe_r = 1_000_000
                xe_r = np.abs(xe_r)
                err_r = np.max(xe_r)
                xe_a = x[k + 1, :] - x[k, :]
                xe_a = np.abs(xe_a)
                err_a = np.max(xe_a)
                error = np.vstack([error, [err_a, err_r]])
                k += 1
                iterations = k
    else:
        print("Error. Enter a number for your selection.")
    return x, error, iterations


def relaxation_method(a, b):
    global error, err_r, err_a, x
    n = a.shape[1]
    error_message = "Error. Enter a number for your selection."
    case = input("Input iterations (1) or convergence condition (2)? ")
    if case == "1":
        iterations = int(input("Enter number of iterations: "))
        converge = "N/A"
        e = "N/A"
    elif case == "2":
        converge = input("Input relative error (1) or absolute error (2)? ")
        e = float(input("Enter convergence condition as a decimal: "))
        iterations = "N/A"
    else:
        print(error_message)

    initial = input("Input an initial guess (1) or start with zero vector (2)? ")
    if initial == "1":
        guess = input("Enter initial guess, comma separated in brackets ([x1,x2,x3,...,xn]): ")
        x = np.fromstring(guess[1:-1], sep=',')
        x = x[np.newaxis, :]
        if x.size == n:
            print("Input accepted")
        elif x.size < n:
            print("Error. Not enough guess inputs.")
        elif x.size > n:
            print("Error. Too many guess inputs.")
        else:
            print("Error. Try again - format guess comma separated in brackets ([x1,x2,x3,...,xn])")
    elif initial == "2":
        x = np.zeros([1, n])
    else:
        print("Error. Enter 'y' or 'n'.")
    omega = float(input("Enter omega (0 < omega < 2): "))

    def iterate(i, k):
        sigma = 0
        for j in range(n):
            if j < i:
                sigma = sigma + a[i, j] * x_k1[j]
            elif j > i:
                sigma = sigma + a[i, j] * x[k, j]
            else:
                sigma = sigma
        x_ijk = (omega / a[i, i]) * (b[i] - sigma) + (1 - omega) * x[k, i]
        return x_ijk

    if case == "1":
        error = np.array([[0, 0]])
        for k in range(iterations):
            x_k1 = np.zeros(n)
            for i in range(n):
                x_k1[i] = iterate(i, k)
            x = np.vstack([x, x_k1])
            if x[k, :].all() != 0:
                xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
            else:
                xe_r = 1_000_000
            xe_r = np.abs(xe_r)
            err_r = np.max(xe_r) * 100
            xe_a = x[k + 1, :] - x[k, :]
            xe_a = np.abs(xe_a)
            err_a = np.max(xe_a)
            error = np.vstack([error, [err_a, err_r]])
    elif case == "2":
        err_r = 1_000_000
        err_a = 1_000_000
        error = np.array([[0, 0]])
        if converge == "2":
            k = 0
            while abs(err_a) > e:
                x_k1 = np.zeros(n)
                for i in range(n):
                    x_k1[i] = iterate(i, k)
                x = np.vstack([x, x_k1])
                xe_a = x[k + 1, :] - x[k, :]
                xe_a = np.abs(xe_a)
                err_a = np.max(xe_a)
                if x[k, :].all() != 0:
                    xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
                else:
                    xe_r = 1_000_000
                xe_r = np.abs(xe_r)
                err_r = np.max(xe_r)
                error = np.vstack([error, [err_a, err_r]])
                k += 1
                iterations = k
        elif converge == "1":
            k = 0
            while abs(err_r) > e:
                x_k1 = np.zeros(n)
                for i in range(n):
                    x_k1[i] = iterate(i, k)
                x = np.vstack([x, x_k1])
                if x[k, :].all() != 0:
                    xe_r = (x[k + 1, :] - x[k, :]) / x[k, :]
                else:
                    xe_r = 1_000_000
                xe_r = np.abs(xe_r)
                err_r = np.max(xe_r)
                xe_a = x[k + 1, :] - x[k, :]
                xe_a = np.abs(xe_a)
                err_a = np.max(xe_a)
                error = np.vstack([error, [err_a, err_r]])
                k += 1
                iterations = k
    else:
        print("Error. Enter a number for your selection.")
    return x, error, iterations

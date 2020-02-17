# This code allows for a system of linear equations to be solved for using any of six methods.
import numpy as np
from numpy import unravel_index
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter as xl
from linear_methods import *

# Data Input
a_manual = np.array([[0, 0.46663158],
                     [0.963, 0.60493882]], dtype=float)
b_manual = np.array([-2.12238731 * 10 ** -7, -1.815], dtype=float)

error_message = "Error. Enter a number for your selection."

data_input = input("Enter data manually (1) or through Excel file import (2)? ")
if data_input == "1":
    a = a_manual.copy()
    b = b_manual.copy()
    b = b[:, None]
elif data_input == "2":
    file = pd.read_excel('input.xls', 0)
    file = np.array(file)
    a = np.array(file[:, 0:-1], dtype=float)
    b = np.array(file[:, -1], dtype=float)
    b = b[:, None]
else:
    print(error_message)

# Pivoting
pivoting = input("Apply partial pivoting (1), complete pivoting (2), or none (0)? ")
n = a.shape[0]


def row_swap(matrix, size):
    for column in range(size):
        a_new = np.abs(a[column:size, :])
        max_index = np.argmax(a_new, axis=0)
        a[[column, max_index[column] + column]] = a[[max_index[column] + column, column]]
        b[column], b[max_index[column] + column] = b[max_index[column] + column], b[column]


def all_swap(matrix, size):
    o = np.arange(n) + 1
    for diagonal in range(size):
        a_new = np.abs(a[diagonal:size, diagonal:size])
        max_value = a_new.argmax()
        max_index = unravel_index(a_new.argmax(), a_new.shape)
        max_index_0, max_index_1 = max_index[0], max_index[1]
        a[:, [diagonal, max_index_1 + diagonal]] = a[:, [max_index_1 + diagonal, diagonal]]
        a[[diagonal, max_index_0 + diagonal]] = a[[max_index_0 + diagonal, diagonal]]
        b[[diagonal, max_index_0 + diagonal]] = b[[max_index_0 + diagonal, diagonal]]
        o[diagonal], o[max_index_0 + diagonal] = o[max_index_0 + diagonal], o[diagonal]
    print(o)
    return o


if pivoting == "1":
    row_swap(a, n)
elif pivoting == "2":
    order = all_swap(a, n)
elif pivoting == "0":
    print("No pivoting applied.")
else:
    print("Error. Try again - enter 'p', 'c', or 'n'.")

# Run Algorithm
method = input("Choose a method:\n"
               "(1) Gauss Elimination\n"
               "(2) Gauss-Jordan Elimination\n"
               "(3) Choleski's Method for Symmetric Matrices\n"
               "(4) Jacobi Iteration Method\n"
               "(5) Gauss-Seidel Method\n"
               "(6) Relaxation Method\n")
if method == "1":
    x = gauss_elimination(a, b)
elif method == "2":
    x = gauss_jordan_elimination(a, b)
elif method == "3":
    x = choleski_method(a, b)
elif method == "4":
    x, error, iterations = jacobi_iteration_method(a, b)
elif method == "5":
    x, error, iterations = gauss_seidel_iteration_method(a, b)
elif method == "6":
    x, error, iterations = relaxation_method(a, b)
else:
    print(error_message)

# Plot, Export, and Print Results
if method == "1" or method == "2" or method == "3":
    print("x = " + str(x))
    if pivoting == "2":
        print("The new order of solutions is: ")
        print(order)

elif method == "4" or method == "5" or method == "6":
    # Display Results and Print to Excel
    print(x)
    print(error)
    print("The absolute error is " + str(error[iterations, 0]))
    print("The relative error is " + str(error[iterations, 1]) + " %")
    print("The number of iterations is " + str(iterations))
    if pivoting == "2":
        print("The new order of solutions is: ")
        print(order)
    file_out = xl.Workbook("Output.xlsx")
    sheet_out = file_out.add_worksheet()
    sheet_error = file_out.add_worksheet('Error')
    col = 0
    for row, dat in enumerate(x):
        sheet_out.write_row(row, col, dat)
    col = 0
    for row, dat in enumerate(error):
        sheet_error.write_row(row, col, dat)
    file_out.close()

    # Plot Convergence History
    h = range(iterations + 1)

    plt.plot(h, x[:, 1], label='x1', color='b')
    plt.plot(h, x[:, 3], label='x2', color='g')
    plt.plot(h, x[:, 0], label='x3', color='r')
    plt.plot(h, x[:, 2], label='x4', color='m')

    plt.xlabel('Iterations, k')
    plt.ylabel('Solution, $x_i$')
    if method == "4":
        plt.title('Jacobi Iteration Method Solution Convergence')
    elif method == "5":
        plt.title('Gauss-Seidel Iteration Method Solution Convergence')
    elif method == "6":
        plt.title('Relaxation Method Solution Convergence')
    else:
        plt.title()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.show()
else:
    print(error_message)

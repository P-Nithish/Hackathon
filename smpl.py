import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = np.array([i for i in range(1, 17)])
an = np.array([3, 6, 11, 21, 32, 47, 65, 87, 112, 110, 171, 204, 241, 282, 325, 376])

delan = [an[0]]
for i in range(1, len(an)):
    delan.append(an[i] - an[i-1])

plt.plot(n, an, label='Stopping Distance')
plt.scatter(n, an, color='red')
plt.title('Stopping Distance vs Speed')
plt.xlabel('Speed (n)')
plt.ylabel('Distance (an)')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(n, delan, label='∆an')
plt.scatter(n, delan, color='orange')
plt.title('First Differences (∆an) vs Speed')
plt.xlabel('Speed (n)')
plt.ylabel('∆an')
plt.grid(True)
plt.legend()
plt.show()

def newton_divided_diff(x, y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])
    return F

def newton_polynomial(x, y, xi):
    F = newton_divided_diff(x, y)
    n = len(x)
    result = F[0, 0]
    product = 1
    for i in range(1, n):
        product *= (xi - x[i - 1])
        result += F[0, i] * product
    return result

F = newton_divided_diff(n, an)
columns = ['f(x)'] + [f'fa  [xi, xi+{i+1}]' for i in range(len(n) - 1)]
div_diff_table = pd.DataFrame(F, index=[f'x{i}' for i in range(len(n))], columns=columns)
print("\nNewton's Divided Difference Table:")
print(div_diff_table)

full_coeffs = F[0, :]
poly_expr = ""
product_term = ""
for i in range(len(full_coeffs)):
    if i > 0:
        product_term += f"(x - {n[i-1]})"
        poly_expr += f" + ({full_coeffs[i]:.4f})*({product_term})"
    else:
        poly_expr += f"({full_coeffs[i]:.4f})"

predicted_an = np.array([newton_polynomial(n, an, x) for x in n])

errors = an - predicted_an
rmse = np.sqrt(np.mean(errors ** 2))

plt.figure(figsize=(10, 5))
plt.plot(n, an, 'bo-', label='Actual Stopping Distance')
plt.plot(n, predicted_an, 'r*--', label='Predicted (Newton Polynomial)')
plt.xlabel("Speed (n, mph)")
plt.ylabel("Stopping Distance (ft)")
plt.title("Actual vs Predicted Stopping Distance (Newton Polynomial)")
plt.grid(True)
plt.legend()
plt.show()

print("\nFull Newton Polynomial Expression (Symbolic Form):")
print("f(x) ≈", poly_expr)
print(f"\nRMSE (Error): {rmse:.4f} ft")
print("\nActual vs Predicted Stopping Distances:")
for i in range(len(n)):
    print(f"Speed = {n[i]} mph, Actual = {an[i]} ft, Predicted = {predicted_an[i]:.2f} ft, Error = {errors[i]:.2f} ft")

quad_coeffs = np.polyfit(n, an, 2)
quad_poly = np.poly1d(quad_coeffs)
quad_predicted = quad_poly(n)
quad_rmse = np.sqrt(np.mean((an - quad_predicted) ** 2))

plt.figure(figsize=(10, 5))
plt.plot(n, an, 'bo-', label='Actual Stopping Distance')
plt.plot(n, quad_predicted, 'g^--', label='Quadratic Model')
plt.xlabel("Speed (n, mph)")
plt.ylabel("Stopping Distance (ft)")
plt.title("Quadratic Fit vs Actual Data")
plt.grid(True)
plt.legend()
plt.show()

----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import math

force = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
stretch = np.array([19, 57, 94, 134, 173, 216, 256, 297, 343])
h = force[1] - force[0]

def newton_forward_diff_table(x, y):
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
    return diff_table

def newton_forward_interpolation(x, x_values, diff_table, h):
    u = (x - x_values[0]) / h
    result = diff_table[0][0]
    u_term = 1
    for i in range(1, len(x_values)):
        u_term *= (u - (i - 1))
        result += (u_term * diff_table[0][i]) / math.factorial(i)
    return result

diff_table = newton_forward_diff_table(force, stretch)
interpolated_points = [15, 17, 85]
interpolated_stretch = [newton_forward_interpolation(x, force, diff_table, h) for x in interpolated_points]

print("Interpolated Stretch Values:")
for x, y in zip(interpolated_points, interpolated_stretch):
    print(f"Force = {x}: Stretch ≈ {y:.2f}")

x_plot = np.linspace(10, 90, 500)
y_plot = [newton_forward_interpolation(xi, force, diff_table, h) for xi in x_plot]

plt.figure(figsize=(8, 6))
plt.plot(force, stretch, 'bo-', label='Original Data')
plt.plot(x_plot, y_plot, 'r--', label='Interpolated Polynomial')
plt.scatter(interpolated_points, interpolated_stretch, color='green', label='Interpolated Values')
plt.xlabel('Force')
plt.ylabel('Stretch')
plt.title('Newton Forward Interpolation')
plt.legend()
plt.grid(True)
plt.show()

predicted_stretch = [newton_forward_interpolation(x, force, diff_table, h) for x in force]
errors = stretch - predicted_stretch

print("\nErrors at Known Points (Actual - Predicted):")
for x, err in zip(force, errors):
    print(f"Force = {x}: Error ≈ {err:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(force, errors, 'm*-', label='Error (Actual - Predicted)')
plt.xlabel('Force')
plt.ylabel('Error')
plt.title('Prediction Error at Known Points')
plt.grid(True)
plt.legend()
plt.show()

np.random.seed(42)
spring_constants = np.random.uniform(1, 5, 5)
simulated_data = [k * force + np.random.normal(0, 5, size=force.shape) for k in spring_constants]

print("\nT-test Results for Simulated Data vs. Original Data:")
for i, sim in enumerate(simulated_data):
    t_stat, p_val = ttest_ind(stretch, sim)
    print(f"Simulation {i + 1} (Spring Constant ≈ {spring_constants[i]:.2f}):")
    print(f"  t-statistic ≈ {t_stat:.2f}, p-value ≈ {p_val:.3f}")


-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

T = np.array([300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
e = np.array([0.024, 0.035, 0.046, 0.058, 0.067, 0.083, 0.097, 0.111, 0.125, 0.140, 0.155, 0.170, 0.186, 0.202, 0.219, 0.235, 0.252, 0.269])

def original_formula(T):
    return 0.02424 * (T / 303.16) ** 1.27591

def newton_divided_diff(x, y):
    n = len(y)
    coef = np.zeros(n)
    coef[0] = y[0]
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
        coef[j] = table[0, j]
    return coef

def newton_eval(x, x_data, coef):
    result = coef[0]
    term = 1
    for i in range(1, len(coef)):
        term *= (x - x_data[i - 1])
        result += coef[i] * term
    return result

newton_coefs = newton_divided_diff(T, e)
newton_values = [newton_eval(x, T, newton_coefs) for x in [0.5, 3]]

def lagrange_interpolation(x, x_data, y_data):
    n = len(x_data)
    result = 0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if j != i:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

lagrange_values = [lagrange_interpolation(x, T, e) for x in [0.5, 3]]

print("Newton Divided Difference Interpolation:")
for x, val in zip([0.5, 3], newton_values):
    print(f"T = {x}: e ≈ {val:.6f}")

print("\nLagrange Interpolation:")
for x, val in zip([0.5, 3], lagrange_values):
    print(f"T = {x}: e ≈ {val:.6f}")

print("\nOriginal Formula Values:")
for x in [0.5, 3]:
    val = original_formula(x)
    print(f"T = {x}: e ≈ {val:.6f}")

x_plot = np.linspace(300, 2000, 500)
newton_plot = [newton_eval(xi, T, newton_coefs) for xi in x_plot]
original_plot = [original_formula(xi) for xi in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(T, e, 'bo', label='Given Data Points')
plt.plot(x_plot, newton_plot, 'r-', label='Newton Interpolating Polynomial')
plt.plot(x_plot, original_plot, 'g--', label='Original Formula')
plt.xlabel('Temperature (K)')
plt.ylabel('Emittance')
plt.title('Emittance vs Temperature')
plt.legend()
plt.grid(True)
plt.show()

newton_at_data = [newton_eval(xi, T, newton_coefs) for xi in T]
errors = e - newton_at_data
print("\nErrors at Data Points (Actual - Interpolated):")
for t, err in zip(T, errors):
    print(f"T = {t}: Error ≈ {err:.6f}")

original_at_data = [original_formula(t) for t in T]
print("\nOriginal Formula vs. Actual Data:")
for t, actual, orig in zip(T, e, original_at_data):
    print(f"T = {t}: Actual e = {actual:.3f}, Formula e ≈ {orig:.3f}, Difference ≈ {actual - orig:.6f}")


----------------------------------------------------------------------


# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 06:53:44 2025

@author: nithi
"""

# -*- coding: utf-8 -*-
"""MML presentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FEAlBxeEHTuVPb3ndLzKwqh3AfeyiIBn
"""

import random
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

"""
n = 20
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
f = [3, 4, 7, 1, 5, 9, 2, 6, 3, 8,4, 7, 0, 5, 6, 2, 4, 1, 9, 3]
"""

n = 4
x = [3,4.5,7,9]
f = [2.5,1,2.5,0.5]

def spline_equation_row(xi_minus1, xi, xi_plus1, f_xi_minus1, f_xi, f_xi_plus1):
    a = xi - xi_minus1
    b = 2 * (xi_plus1 - xi_minus1)
    c = xi_plus1 - xi
    rhs = (6 / (xi_plus1 - xi)) * (f_xi_plus1 - f_xi) - (6 / (xi - xi_minus1)) * (f_xi - f_xi_minus1)
    return a, b, c, rhs

variables = sp.symbols(f'f\'\'1:{n-1}')
equations = []
print("--------",variables)
for i in range(1, n - 1):
    a, b, c, rhs = spline_equation_row(x[i - 1], x[i], x[i + 1], f[i - 1], f[i], f[i + 1])
    row = 0
    if i > 1:
        row += a * variables[i - 2]
    print("***",row)
    row += b * variables[i - 1]
    print("*****",row)
    if i < n - 2:
        row += c * variables[i]
    print("*****",row)
    equations.append(sp.Eq(row, rhs))
    print(equations,"-=-=-=-=")

sol = sp.solve(equations, variables)
print(sol,'----====')
fpp = [0] + [sol[v] for v in variables] + [0]

def cubic_spline_interpolation(x_val, xi, xi1, fxi, fxi1, fppi, fppi1):
    h = xi1 - xi
    term1 = fppi * ((xi1 - x_val) ** 3) / (6 * h)
    term2 = fppi1 * ((x_val - xi) ** 3) / (6 * h)
    term3 = (fxi / h - fppi * h / 6) * (xi1 - x_val)
    term4 = (fxi1 / h - fppi1 * h / 6) * (x_val - xi)
    print(f"{term1}+{term2}+{term3}+{term4}")
    return term1 + term2 + term3 + term4

X_vals = []
Y_vals = []

for i in range(n - 1):
    xs = np.linspace(x[i],x[i+1],100)
    ys = [cubic_spline_interpolation(xv, x[i], x[i + 1], f[i], f[i + 1], fpp[i], fpp[i + 1]) for xv in xs]
    X_vals.extend(xs)
    Y_vals.extend(ys)


plt.figure(figsize=(12, 6))
plt.plot(X_vals, Y_vals, label="Cubic Spline", color="blue")
plt.plot(x, f, 'ro', label="Data Points")
plt.title("Cubic Spline Interpolation")
plt.legend()
plt.grid(True)
plt.show()

interpolated_y_vals=[]
x_vals=[5,4.5]

for x_val in x_vals:
  for i in range(n - 1):
      if x[i] <= x_val <= x[i + 1]:
          y_val = cubic_spline_interpolation(
              x_val, x[i], x[i + 1],
              f[i], f[i + 1],
              fpp[i], fpp[i + 1]
          )
          interpolated_y_vals.append(y_val)
          print(f"interpolated value at x = {x_val} is {y_val}")
          break

print("System of Linear Equations for f'' values (formatted):\n")
for eq in equations:
    lhs = sp.simplify(eq.lhs)
    rhs = sp.simplify(eq.rhs)
    print(f"{lhs} = {rhs}")

print('\n\nSolutions of the above equations are :- \n')
for i in range(0, n):
    print(f"f''{i} = {fpp[i]}")

x_sym = sp.Symbol('x')
spline_functions = []

print("Cubic Spline Functions for Each Interval:\n")

for i in range(n - 1):
    xi = x[i]
    xi1 = x[i + 1]
    h = xi1 - xi
    fxi = f[i]
    fxi1 = f[i + 1]
    fppi = fpp[i]
    fppi1 = fpp[i + 1]

    term1 = fppi * ((xi1 - x_sym) ** 3) / (6 * h)
    term2 = fppi1 * ((x_sym - xi) ** 3) / (6 * h)
    term3 = (fxi / h - fppi * h / 6) * (xi1 - x_sym)
    term4 = (fxi1 / h - fppi1 * h / 6) * (x_sym - xi)

    Si = sp.simplify(term1 + term2 + term3 + term4)
    spline_functions.append(Si)

    print(f"S{i}(x) for interval [{xi}, {xi1}]:")
    print(Si)
    print()


----------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
import math

def standard_normal_pdf(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / (n - 1)
    res = f(a)
    for i in range(1, n-1):
      a += h
      res += 2 * f(a)
    res += f(b)
    return (h / 2) * res

def simpsons_rule(f, a, b, n):
  #Note that the method can be employed only if the number of segments is even.
    if n % 2 == 0:
        n += 1
    h = (b - a) / (n - 1)
    result = f(a) + f(b)
    for i in range(1, n - 1):
        a += h
        if i % 2 == 0:
            result += 2 * f(a)
        else:
            result += 4 * f(a)
    return (h / 3) * result

a, b = -4, 2
true_value = 0.977249868051821
points_list = [2001, 4001]

trap_results = []
simp_results = []

for n in points_list:
    trap = trapezoidal_rule(standard_normal_pdf, a, b, n)
    simp = simpsons_rule(standard_normal_pdf, a, b, n)
    trap_results.append(trap)
    simp_results.append(simp)
    print(f"Points = {n}")
    print(f"  Trapezoidal Rule: {trap:.15f}, Error: {abs(trap - true_value):.10f}")
    print(f"  Simpson's Rule  : {simp:.15f}, Error: {abs(simp - true_value):.10f}")
    print()

    x_vals = np.linspace(a, b, n)
    y_vals = [standard_normal_pdf(x) for x in x_vals]
    original = np.linspace(a, b, 5000)
    original_y = [standard_normal_pdf(x) for x in original]


---------------------------------------------------------------


from sympy import symbols, Function, dsolve, Eq, simplify, expand, lambdify, solve
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.stats import ttest_ind

# Initial values
x0 = 0
y0 = 1.0
h = 0.1
n = 15

# f(x) = dy/dx
def f(x,y):
    return -2*y + x**2

# Euler's Method
def euler_method(x0, y0, h, n):
    x_vals = [x0]
    y_vals = [y0]
    for i in range(n):
        y_next = y_vals[-1] + h * f(x_vals[-1],y_vals[-1])
        x_next = x_vals[-1] + h
        x_vals.append(x_next)
        y_vals.append(y_next)
    return x_vals, y_vals

x = symbols('x')
y = Function('y')
ode = Eq(y(x).diff(x), -2*y(x) + x**2)
sol = dsolve(ode, y(x))
print(sol)

# Solve for C1 manually
C1 = symbols('C1')
rhs_expr = sol.rhs
print(rhs_expr)
eq_C1 = Eq(rhs_expr.subs(x, x0), y0)
print(eq_C1)
C1_vals = solve(eq_C1, C1)

# Pick valid real positive C1
C1_val = [c for c in C1_vals if c.is_real and c > 0][0]

# Get particular solution
sol_expr = rhs_expr.subs(C1, C1_val)
print(sol_expr)
sol_expr = simplify(expand(sol_expr))
print(sol_expr)

# Prepare numerical function
f_analytical = lambdify(x, sol_expr, 'numpy')

print("="*50)
print("\tGiven Data")
print("="*50)
print("\tx0 : ",x0)
print("\ty0 : ",y0)
print("\tStep size : ",h)
print("\tf'(x) : -2*y + x**2")
print("="*50)

print("\nAnalytical Solution:")
print("f(x) = ", sol_expr)

x_values = [x0 + i*h for i in range(n+1)]
y_true = f_analytical(np.array(x_values))

x_euler, y_euler = euler_method(x0, y0, h, n)

table = PrettyTable()
table.field_names = ["x", "Analytical", "Euler"]
for i in range(n+1):
    table.add_row([
        round(x_values[i], 3),
        round(y_true[i], 5),
        round(y_euler[i], 5),
    ])
print()
print(table)

# T-tests
t_euler = ttest_ind(y_true, y_euler)

print("\nT-test Interpretation:")
print("- Euler's Method: p-value =", t_euler.pvalue,"->", "Significant difference" if t_euler.pvalue < 0.05 else "No significant difference")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x_values, y_true, label="Analytical", linewidth=2)
plt.plot(x_euler, y_euler, '--', label="Euler", alpha=0.7)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Numerical Methods with Analytical Solution")
plt.grid(True)
plt.tight_layout()
plt.show()
------------------------------------


from sympy import symbols, Function, dsolve, Eq, simplify, expand, lambdify
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.stats import ttest_ind

# Initial values
x0 = 0
y0 = 1.0
h = 0.1
n = 15

# f(x) = dy/dx
def f(x,y):
    return -2*y + x**2

# Runge-Kutta 4th Order Method
def rk4_method(x0, y0, h, n):
    x_vals = [x0]
    y_vals = [y0]
    for i in range(n):
        x = x_vals[-1]
        y = y_vals[-1]
        k1 = f(x,y)
        k2 = f(x + h/2,y + (k1*h/2))
        k3 = f(x + h/2,y + (k2*h/2))
        k4 = f(x + h,y + k3*h)
        y_next = y + ((k1 + 2*k2 + 2*k3 + k4)*h)/6
        x_next = x + h
        x_vals.append(x_next)
        y_vals.append(y_next)
    return x_vals, y_vals

x = symbols('x')
y = Function('y')
ode = Eq(y(x).diff(x), -2*y(x) + x**2)
sol = dsolve(ode, y(x), ics={y(x0): y0})
sol_expr = simplify(expand(sol.rhs))
f_analytical = lambdify(x, sol_expr, 'numpy')

print("="*50)
print("\tGiven Data")
print("="*50)
print("\tx0 : ",x0)
print("\ty0 : ",y0)
print("\tStep size : ",h)
print("\tf'(x) : -2*y + x**2")
print("="*50)

print("\nAnalytical Solution:")
print("f(x) = ", sol_expr)

x_values = [x0 + i*h for i in range(n+1)]
y_true = f_analytical(np.array(x_values))

x_rk4, y_rk4 = rk4_method(x0, y0, h, n)

table = PrettyTable()
table.field_names = ["x", "Analytical", "RK4"]
for i in range(n+1):
    table.add_row([
        round(x_values[i], 3),
        round(y_true[i], 5),
        round(y_rk4[i], 5),
    ])
print()
print(table)

# T-tests
t_rk4 = ttest_ind(y_true, y_rk4)

print("\nT-test Interpretation:")
print("- RK4: p-value =", t_rk4.pvalue,"->", "Significant difference" if t_rk4.pvalue < 0.05 else "No significant difference")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x_values, y_true, label="Analytical", linewidth=2)
plt.plot(x_rk4, y_rk4, '--', label="RK4", alpha=0.7)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Numerical Methods with Analytical Solution")
plt.grid(True)
plt.tight_layout()
plt.show()
----------------

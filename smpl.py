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

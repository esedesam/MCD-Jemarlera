import sympy as sp
import numpy as np

# Armijo para obtener lambda
def armijo_line_search(func, gradient, xk, direction, lamb = 0.5, c1 = 0.9, max_iter = 100):
    """
    Armijo line search to find a suitable step size.

    Parameters:
    - func: The objective function to minimize.
    - gradient: The gradient of the objective function.
    - x: Current iterate.
    - direction: Search direction.
    - lamb: Initial step size.
    - c1: Armijo condition parameter.
    - max_iter: Maximum number of iterations.

    Returns:
    - lamb: The step size that satisfies Armijo-Wolfe conditions.
    """
    def armijo_condition(lamb):
        x_new = xk + lamb * direction
        func_value = func(*x_new)  # Evaluate the symbolic function at x_new
        return (func_value <= func(*xk) + c1 * lamb * gradient.dot(direction))
    
    for _ in range(max_iter):
        if armijo_condition(lamb):
            break
        lamb = lamb / 2  # Reduce step size if conditions are not met

    return lamb  # Return the best step size found

# Test de convergencia de descenso lineal
def test_convergence_linear_descend(x0, x_new, func, gradient_point_new, tol1, tol2, tol3):

    conditions = (np.linalg.norm(x0 - x_new) < tol1,
                  np.linalg.norm(gradient_point_new) < tol2,
                  abs(func(*x_new) - func(*x0)) < tol3)
    
    test = any(conditions)
    condition_idx = np.where(conditions)[0]

    return test, condition_idx

# Método del gradiente
def gradient_linear_descend(f, x, x0, tol1 = 1e-4, tol2 = 1e-4, tol3 = 1e-4, max_iterations = 100):
    
    func = sp.lambdify(x, f, 'numpy')

    gradient = [sp.diff(f, xi) for xi in x]
    gradient_func = sp.lambdify(x, gradient, 'numpy')
    
    gradient_point = gradient_func(*x0)
    gradient_point = np.array(gradient_point)
    d0 = -gradient_point
    
    # Inicialización de variables para el bucle
    iteration = 0
    
    while iteration < max_iterations:
        
        lamb = armijo_line_search(func, gradient_point, x0, d0)
        
        x_new = x0 + lamb * d0 # Aquí se obtiene el x_{k + 1}
        gradient_point_new = np.array(gradient_func(*x_new))
        d_new = -gradient_point_new
        
        test, cond_idx = test_convergence_linear_descend(x0, x_new, func, gradient_point_new, tol1, tol2, tol3)
        if test:
            print(f"Convergence reached in iteration {iteration}. Fullfilled conditions: {cond_idx}")
            break
        else:
            x0 = x_new
            d0 = d_new
            gradient_point = gradient_point_new
            iteration = iteration + 1
    
    return x_new, iteration
    
# Llamada a función

x, y = sp.symbols('x y')
f = x - y + 2*x**2 + 2*x*y + y**2

initial_guess = [0, 1]

result, iterations = gradient_linear_descend(f, (x, y), initial_guess)
print(result)
print(iterations)
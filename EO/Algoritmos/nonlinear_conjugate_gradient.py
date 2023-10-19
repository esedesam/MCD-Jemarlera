import sympy as sp
import numpy as np

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

def fletcher_reeves(f, x, x0, tol = 1e-4, max_iterations = 100):
    
    gradient = [sp.diff(f, xi) for xi in x]
    func = sp.lambdify(x, f, 'numpy')
    gradient_func = sp.lambdify(x, gradient, 'numpy')
    
    gradient_point = gradient_func(*x0)
    gradient_point = np.array(gradient_point)
    d0 = -gradient_point
    
    # Inicialización de variables para el bucle
    iteration = 0
    x_new = x0
    d_new = d0
    gradient_point_new = gradient_point
    
    while (np.linalg.norm(gradient_point) > tol and iteration < max_iterations):
        
        lamb = armijo_line_search(func, gradient_point_new, x_new, d_new)
        
        gradient_point_old = gradient_func(*x_new) # Este es el gradiente evaluado en el x_k
        
        x_new = x_new + lamb*d_new # Aquí se pone el x_{k + 1}
        
        gradient_point_new = gradient_func(*x_new)
        gradient_point_new = np.array(gradient_point_new)
        
        beta_new = np.dot(gradient_point_new, gradient_point_new) / np.dot(gradient_point_old, gradient_point_old)
        d_new = -gradient_point_new + beta_new * d_new
        
        iteration = iteration + 1
    
    return x_new, iteration
    
# Llamada a función

x, y = sp.symbols('x y')
f = x - y + 2*x**2 + 2*x*y + y**2

initial_guess = [0, 1]

result, iterations = fletcher_reeves(f, (x, y), initial_guess)
print(result)
print(iterations)
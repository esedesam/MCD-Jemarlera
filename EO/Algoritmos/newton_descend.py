import sympy as sp
import numpy as np

def newtons_method(f, x, initial_guess, tol = 1e-6, max_iter = 100):
    
    """
    Newton's method for finding the minimum of a multivariable function.
    
    Parameters:
    f: function to minimize
    x: list of sympy symbols
    initial_guess: initial point to iterate
    tol: float, optional
        Tolerance for stopping criterion. The iteration stops when the norm
        of the gradient is less than tol.
    max_iter: int, optional
        Maximum number of iterations.
        
    Returns:
    x_0: array
        The estimated minimum point as a list of numerical values.
    iteration: int
        The number of iterations performed.
    """
    gradient = [sp.diff(f, xi) for xi in x]
    hessian = sp.hessian(f, x)
    gradient_func = sp.lambdify(x, gradient, 'numpy')
    hessian_func = sp.lambdify(x, hessian, 'numpy')

    x0 = initial_guess
    
    for iteration in range(max_iter):
        grad = gradient_func(*x0)
        hess = hessian_func(*x0)
        step = -np.dot(np.linalg.inv(hess), grad)
        x0 = x0 + step
        
        if np.linalg.norm(grad) < tol:
            break

    return x0, iteration
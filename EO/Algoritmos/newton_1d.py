import sympy as sp

def newton_optimization_1d(f, x0, tol = 1e-6, max_iter = 100):
    
    x = sp.symbols('x')
    
    f_expr = sp.sympify(f)
    
    f_prime = f_expr.diff(x)
    f_double_prime = f_prime.diff(x)
    
    x_current = x0
    iteration = 0
    
    while iteration < max_iter:
        df_prime = f_prime.subs(x, x_current)
        df_double_prime = f_double_prime.subs(x, x_current)
        
        x_new = x_current - df_prime / df_double_prime # iteración
        
        if abs(x_new - x_current) < tol: # salida por tolerancia
            return x_new
        
        x_current = x_new
        iteration += 1
        
    return x_current
import numpy as np

def conjugate_gradient(A, b, x0, tol = 1e-6, max_iter = 100):
    
    x = x0
    r = np.dot(A, x) - b 
    d = -r
    
    for iteration in range(max_iter):
        Ad = np.dot(A, d)
        lamb = np.dot(r, r) / np.dot(d, Ad)
        x = x + lamb * d
        r_new = np.dot(A, x) - b
        
        if np.linalg.norm(r_new) < tol:
            break
        
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        d = -r_new + beta * d
        r = r_new
    return x, iteration
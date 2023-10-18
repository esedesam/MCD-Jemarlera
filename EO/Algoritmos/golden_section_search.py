import math

def golden_ratio_search(func, lower, upper, tolerance):

    golden_ratio = (math.sqrt(5) - 1) / 2

    # Calculate initial values for the search
    a = lower
    b = upper
    x1 = a + (1 - golden_ratio) * (b - a)
    x2 = a + golden_ratio * (b - a)

    while abs(b - a) > tolerance:
        if func(x1) < func(x2):
            b = x2 # a no se ve modificado
        else:
            a = x1 # b no se ve modificado

        x1 = a + (1 - golden_ratio) * (b - a)
        x2 = a + golden_ratio * (b - a)

    # Return the approximate minimum point and value
    min_point = (a + b) / 2
    min_value = func(min_point)

    return min_point, min_value
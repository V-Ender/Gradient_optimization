import numpy as np

'''
Все реализованные здесь функции представляют собой
проблемные для оптимизации функции.
Взяты с сайта:
https://www.sfu.ca/~ssurjano/optimization.html
'''


def ackley_f(x):
    '''
    Hole in the center with many local minima around
    bounds: 32
    '''
    a = 20
    b = 0.2
    c = 2 * np.pi

    return -a * np.exp(-b * np.sqrt((x**2).mean())) - np.exp(np.cos(c * x).mean()) + a + np.exp(1)


def buckin_6(x):
    '''
    2d with many local minima on one ridge
    bounds: 5
    '''

    return 100 * (np.sqrt(np.abs(x[1] - 0.01 * x[1]**2))) + 0.01 * np.abs(x[0] + 10)


'''def cross_in_tray(x):
    
    2d with multiple global and local minima
    bounds: 10
    minima: -2.06261 at x = (+-1.3491, +-1.3491)
    

    return -0.0001 * ()'''


def drop_wave(x):
    '''
    2d multimodal and highly complex
    bounds: 5.12
    minima: -1 at x = 0
    '''

    return - (1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))) / (0.5 * (x[0]**2 + x[1]**2 ) + 2)


def griewank(x):
    '''
    d-dim with many widespread local minima
    bounds: 600
    minima: 0 at 0
    '''
    arr = np.array(range(len(x))) + 1

    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(arr))) + 1


def rastrigin(x):
    '''
    d-dim highly multimodal with several local minima
    bounds: 5.12
    minima: 0 at 0
    '''

    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def schaffer2(x):
    '''
    2-dim with many local minima
    bounds: 100
    minima: 0 at 0
    '''

    return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2


def shubert(x):
    '''
    2-dim with several local minima and many gloabal minima
    bounds: 10
    minima: -186.7309
    '''
    arr = np.array(range(5)) + 1

    return (np.sum(arr * np.cos((arr + 1) * x[0] + arr))) * (np.sum(arr * np.cos((arr + 1) * x[1] + arr)))


#-----------------------------
# Bowl-Shaped functions
#-----------------------------

def bohachevsky(x):
    '''
    2d with simple bowl shape
    bounds: 100
    minima: 0 at x = 0
    '''

    return x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7


def perm(x):
    '''
    d-dim function
    bounds: d
    minima: 0 at x = (1, 1/2, ... , 1/d)
    '''
    beta = 0
    s = 0
    for i in range(len(x)):
        s_i = 0
        for j in range(len(x)):
            s_i += (j + beta) * (x[j]**i - 1 / (j**i))
        s += s_i * s_i

    return s


def rot_hyper_ellipsoid(x):
    '''
    d-dim convex and unimodal function
    bounds: 65.536
    minima: 0 at x = 0
    '''

    return np.sum(np.cumsum(x**2))


def sphere(x):
    '''
    d-dim convex and unimodal function
    bounds: 5.12
    minima: 0 at x = 0
    '''

    return np.sum(x**2)


def sum_of_powers(x):
    '''
    d-dim unimodal function
    bounds: 1
    minima: 0 at x = 0
    '''

    s = 0
    for i in range(len(x)):
        s += np.abs(x[i])**(i + 1)

    return s


def sum_squares(x):
    '''
    d-dim unimodal function
    bounds: 10
    minima: 0 at x = 0
    '''
    arr = np.array(range(len(x))) + 1

    return np.sum(arr * x)


def trid(x):
    '''
    d-dim no local minima function except global
    bounds: d**2
    minima: -d(d + 4)(d - 1) / 6 at x_i = i(d + 1 - i)
    '''

    return np.sum(x - 1) - np.sum(x[1:] * x[:-1])


#-----------------------------
# Plate-Shaped functions
#-----------------------------


def booth(x):
    '''
    2-dim function
    bounds: 10
    minima: 0 at x = (1, 3)
    '''

    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)


def matyas(x):
    '''
    2-dim function
    bounds: 10
    minima: 0 at x = 0
    '''

    return 0.26 * np.sum(x**2) - 0.48 * np.prod(x)


def mccormick(x):
    '''
    2-dim function
    bounds: 4
    minima: -1.9133 at x = (-0.54719, -1.54719)
    '''

    return np.sin(np.sum(x)) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1


def power_sum(x):
    '''
    4-dim function
    bounds: 4
    minima: -1.9133 at x = (-0.54719, -1.54719)
    '''
    b = np.array([8, 18, 44, 114])
    s = 0
    for i in range(len(x)):
        s_i = 0
        for j in range(len(x)):
            s_i += x[j]**(i + 1)
        s += (s_i - b[i])**2

    return s


def zakharov(x):
    '''
    d-dim function
    bounds: 10
    minima: 0 at x = 0
    '''
    arr = (np.array(range(len(x))) + 1) * 0.5

    return np.sum(x**2) + np.sum(arr * x)**2 + np.sum(arr * x)**4


#-----------------------------
# Valley-Shaped functions
#-----------------------------


def three_hump_camel(x):
    '''
    2-dim function
    bounds: 5
    minima: 0 at x = 0
    '''

    return 2 * x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2


def six_hump_camel(x):
    '''
    2-dim function
    bounds: 3
    minima: -1.0316 at x = (-0.0898, 0.7126) and x = (0.0898, -0.7126)
    '''

    return (4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2


def dixon_price(x):
    '''
    d-dim function
    bounds: 10
    minima: 0 at x_i = 2**(-(2**i - 2) / 2**i)
    '''
    arr = np.array(range(len(x) - 1)) + 2

    return (x[0] - 1)**2 + np.sum(arr * (2 * x[1:]**2 - x[:-1])**2)


def rosenbrock(x):
    '''
    d-dim function
    bounds: 10
    minima: 0 at x = 1
    '''

    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)


#-----------------------------
# Steep Ridges/Drops functions
#-----------------------------


def de_jong5(x):
    '''
    2-dim function multimodal with sharp drops on flat surface
    bounds: 65.536
    minima:
    '''
    a1 = np.array([-32, -16, 0, 16, 32] * 5)
    a2 = np.array([-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5)
    arr = np.array(range(25)) + 1

    return 1 / (0.002 + np.sum(1 / (arr + (x[0] - a1)**6 + (x[1] - a2)**6)))


def easom(x):
    '''
    2-dim function, global minimum has a small area relative to the search space
    bounds: 100
    minima: -1 at x = (pi, pi)
    '''

    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)


def michalewicz(x):
    '''
    d-dim function with d! local minima
    bounds: pi
    minima: -1 at x = (pi, pi)
    '''
    m = 10 # larger m leads to more difficult search
    arr = np.array(range(len(x))) + 1

    return -np.sum(np.sin(x) * np.sin(arr * x**2 / np.pi)**(2 * m))




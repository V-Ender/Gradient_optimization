import numpy as np
import numdifftools as nd


class GlobalGradientDescent:
    '''
    Gradient descent with 1-d optimization along gradient vector
    '''
    def __init__(self, function, max_iters=10000, learning_rate=0.1, eps=1e-8, dims=2, bound=32, verbose=False, global_opt=None):
        '''

        :param function: function, that is being minimized
        :param max_iters: int, max number of iterations
        :param learning_rate: float, step along gradient vector
        :param eps: float, tolerance level, if norm(x - prev) < eps stop descent
        :param dims: int, number of dimensions of target function
        :param bound: float, size of rectangle with center in 0
        :param verbose: bool, verbose
        '''
        self.f = function
        self.max_iters = max_iters
        self.eps = eps
        self.learning_rate = learning_rate
        self.bound = bound
        self.verbose = verbose
        self.global_opt = global_opt

        self.x = np.zeros(dims)
        self.prev = None
        f0 = self.f(self.x)
        self.fs = []

        '''if verbose:
            print("The cost at x = 0 is: " + str(f0))'''


    def step(self, iteration):
        '''
        Performs one optimization along gradient vector
        :param iteration: number of iterations
        :return:
        '''
        self.prev = np.copy(self.x)
        grad = nd.Gradient(self.f)(self.x)

        p = self.x
        d = grad
        t_min, t_max = self.t_bounds(p, d, self.bound)

        phi = lambda t: p + t * d
        f = lambda t: self.f(phi(t))
        opt = self.optimize(f, float(t_min), float(t_max), self.learning_rate)

        self.x = self.x + opt * d
        f = self.f(self.x)

        if (f > 10000):
            print("Cost explosion, trying with lower learning rate")
            self.learning_rate = self.learning_rate / 10
            print("Learning rate changed to " + str(round(self.learning_rate, 2)))

        if (self.verbose):
            print("The cost is: " + str(f))

        self.fs.append(f)


    def run(self):
        '''
        Runs gradient descent
        :return: (np.array, list), optimal point, list of function values during descent
        '''
        self.x = self.initialize(self.bound)
        for i in range(self.max_iters):
            self.step(i)

            if self.stop(self.eps):
                break

        x_opt = self.x

        return x_opt, self.fs


    def initialize(self, bound):
        '''
        Initialize starting point with random values
        :param bound: float, rectangle bound
        :return: np.array, starting point
        '''
        for i in range(len(self.x)):
            self.x[i] = np.random.uniform(-bound, bound)

        return self.x


    def stop(self, eps):
        '''
        Stopping criteria
        :param eps: float, tolerance level
        :return: bool, is criteria met
        '''
        
        if self.global_opt is not None:
            if abs(self.f(self.x) - self.global_opt) < eps:
                return True

        return False


    def optimize(self, f, lb, rb, step):
        '''
        Performs 1-d optimization
        :param f: function being minimized
        :param lb: float, left bound
        :param rb: float, right bound
        :param step: number of steps between lb and rb
        :return: np.array, minimum point
        '''
        step = self.learning_rate * (rb - lb)
        x_arr = np.arange(lb, rb, step)
        min_x = lb
        for x in x_arr:
            if f(x) < f(min_x):
                min_x = x

        return min_x


    def t_bounds(self, p, d, a):
        '''
        Find minimum and maximum t, so optimized area is inside rectangle
        :param p: np.array, point
        :param d: np.array, vector
        :param a: float, bound
        :return: (float, float), min and max t
        '''
        dim = len(p)
        pos_t, neg_t = np.inf, -np.inf
        for i in range(dim):

            t_tmp = (a - p[i]) / d[i]
            if (t_tmp >= 0) and (t_tmp < pos_t):
                pos_t = t_tmp
            if (t_tmp < 0) and (t_tmp > neg_t):
                neg_t = t_tmp

            t_tmp = (-a - p[i]) / d[i]
            if (t_tmp >= 0) and (t_tmp < pos_t):
                pos_t = t_tmp
            if (t_tmp < 0) and (t_tmp > neg_t):
                neg_t = t_tmp

        return (neg_t, pos_t)


class RandomVectorDescent(GlobalGradientDescent):
    def __init__(self, function, max_iters=10000, learning_rate=0.1, num_vectors=1, eps=1e-8, dims=2, bound=32, verbose=False, global_opt=0):
        '''

        :param function: function, that is being minimized
        :param max_iters: int, max number of iterations
        :param learning_rate: float, step along gradient vector
        :param num_vectors: int, number of random vectors to optimize
        :param eps: float, tolerance level, if norm(x - prev) < eps stop descent
        :param dims: int, number of dimensions of target function
        :param bound: float, size of rectangle with center in 0
        :param verbose: bool, verbose
        '''
        super().__init__(function, max_iters, learning_rate, eps, dims, bound, verbose, global_opt)
        self.num_vectors = num_vectors

        self.x = np.zeros(dims)
        self.prev = None
        self.best_f = np.inf
        self.fs = []

    def step(self, iteration):
        '''
        Performs one optimization along all random vectors
        :param iteration: number of iterations
        :return:
        '''
        self.prev = np.copy(self.x)
        best_x = np.copy(self.x)

        for i in range(self.num_vectors):
            vector = np.random.rand(len(self.x)) * 2 - 1

            p = self.x
            d = vector
            t_min, t_max = self.t_bounds(p, d, self.bound)

            phi = lambda t: p + t * d
            f = lambda t: self.f(phi(t))
            opt = self.optimize(f, float(t_min), float(t_max), self.learning_rate)

            x = self.x + opt * d
            f = self.f(x)
            if f < self.best_f:
                self.best_f = f
                best_x = x

        if (self.best_f > 10000):
            print("Cost explosion, trying with lower learning rate")
            self.learning_rate = self.learning_rate / 10
            print("Learning rate changed to " + str(round(self.learning_rate, 2)))

        if (self.verbose):
            print("The cost is: " + str(self.best_f))

        self.x = best_x
        self.fs.append(self.best_f)
        

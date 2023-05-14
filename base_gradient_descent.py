import numpy as np
import numdifftools as nd


class VanillaGradientDescent:
    '''
    Vanilla gradient descent algorithm
    '''

    def __init__(self, function, max_iters=10000, learning_rate=0.1, eps=1e-8, dims=2, bound=32, verbose=False):
        '''

        :param function: function, that is being minimized
        :param max_iters: int, max number of iterations
        :param learning_rate: float, learning rate
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

        self.x = np.zeros(dims)
        self.prev = None
        f0 = self.f(self.x)
        self.fs = []

        '''if verbose:
            print("The cost is: " + str(f0))'''


    def step(self, iteration):
        '''
        Performs one step of gradient descent
        :param iteration: iteration number
        :return:
        '''
        self.prev = np.copy(self.x)
        grad = nd.Gradient(self.f)(self.x)
        self.x = self.x - self.learning_rate * grad
        f = self.f(self.x)

        if (f > 10000):
            print("Cost explosion, trying with lower learning rate")
            self.learning_rate = self.learning_rate / 10
            print("Learning rate changed to " + str(round(self.learning_rate, 2)))

        if (self.verbose):
            if (iteration % 10 == 0):
                print("The cost is: " + str(f))

        self.fs.append(f)


    def run(self):
        '''
        Runs gradient descent algorithm
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
        if np.linalg.norm(self.x - self.prev) < eps:
            return True

        return False


class MomentumGradientDescent(VanillaGradientDescent):
    '''
    Gradient descent with momentum
    '''

    def __init__(self, function, max_iters=10000, learning_rate=0.1, momentum=0.9, eps=1e-8, dims=2, bound=32,
                 verbose=False):
        '''

        :param function: function, that is being minimized
        :param max_iters: int, max number of iterations
        :param learning_rate: float, learning rate
        :param momentum: float, momentum
        :param eps: float, tolerance level, if norm(x - prev) < eps stop descent
        :param dims: int, number of dimensions of target function
        :param bound: float, size of rectangle with center in 0
        :param verbose: bool, verbose
        '''
        super().__init__(function, max_iters, learning_rate, eps, dims, bound, verbose)

        self.momentum = momentum
        self.change = np.zeros(dims)


    def step(self, iteration):
        '''
        Performs one step of gradient descent
        :param iteration: iteration number
        :return:
        '''
        self.prev = np.copy(self.x)
        grad = nd.Gradient(self.f)(self.x)
        new_change = self.learning_rate * grad + self.momentum * self.change
        self.x = self.x - new_change
        self.change = new_change
        f = self.f(self.x)

        if (f > 10000):
            print("Cost explosion, trying with lower learning rate")
            self.learning_rate = self.learning_rate / 10
            print("Learning rate changed to " + str(round(self.learning_rate, 2)))

        if (self.verbose):
            if (iteration % 10 == 0):
                print("The cost is: " + str(f))

        self.fs.append(f)


class AdamGradientDescent(VanillaGradientDescent):
    '''
    Adam gradient descent
    '''
    def __init__(self, function, max_iters=10000, learning_rate=0.1, beta1=0.9, beta2=0.999, v_epsilon=1e-8,
                 eps=1e-5, dims=2, bound=32, verbose=False):
        '''

        :param function: function, that is being minimized
        :param max_iters: int, max number of iterations
        :param learning_rate: float, learning rate
        :param beta1: float, beta1
        :param beta2: float, beta2
        :param v_epsilon: float, value to add in adam denominator for numeric stability
        :param eps: float, tolerance level, if norm(x - prev) < eps stop descent
        :param dims: int, number of dimensions of target function
        :param bound: float, size of rectangle with center in 0
        :param verbose: bool, verbose
        '''
        super().__init__(function, max_iters, learning_rate, eps, dims, bound, verbose)

        self.beta1 = beta1
        self.beta2 = beta2
        self.v_epsilon = v_epsilon

        self.m = np.zeros(dims)
        self.v = np.zeros(dims)


    def step(self, iteration):
        '''
        Performs one step of gradient descent
        :param iteration: iteration number
        :return:
        '''
        self.prev = np.copy(self.x)
        grad = nd.Gradient(self.f)(self.x)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        m = self.m / (1 - self.beta1 ** (iteration + 1))
        v = self.v / (1 - self.beta2 ** (iteration + 1))

        self.x = self.x - self.learning_rate * m / (np.sqrt(v) + self.v_epsilon)
        f = self.f(self.x)

        if (f > 10000):
            print("Cost explosion, trying with lower learning rate")
            self.learning_rate = self.learning_rate / 10
            print("Learning rate changed to " + str(round(self.learning_rate, 2)))

        if (self.verbose):
            if (iteration % 10 == 0):
                print("The cost is: " + str(f))

        self.fs.append(f)

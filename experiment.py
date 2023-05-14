import numpy as np
import matplotlib.pyplot as plt
import time


class Experiment:
    '''
    Class for conducting experiments and plotting results
    '''
    def __init__(self, method, function, num_trials):
        '''

        :param method: optimization method
        :param function: funtiton being optimized
        :param num_trials: int, number of trials
        '''
        self.method = method
        self.function = function
        self.num_trials = num_trials
        self.trial_stats = [None] * num_trials


    def run(self):
        '''
        runs experiments
        :return:
        '''
        for i in range(self.num_trials):
            np.random.seed(i)

            start = time.time()
            x_opt, fs = self.method.run()
            end = time.time()
            self.trial_stats[i] = {'elapsed_time': end - start,
                                   'x_opt': x_opt,
                                   'function_values': fs,
                                   'iterations': len(fs)}
            self.method.clear()


    def plot_trial(self, trial, title=''):
        '''
        plots results of one trial
        :param trial: int, trial number
        :param title: plot title
        :return:
        '''
        plt.plot(self.trial_stats[trial]['function_values'])
        plt.title(title)
        plt.xlabel('Iteration number')
        plt.ylabel('Function value')
        plt.show()

        print(self.trial_stats[trial]['elapsed_time'], self.trial_stats[trial]['iterations'])


    def plot(self, title=''):
        '''
        plots all trials
        :param title: title of one plot
        :return:
        '''
        for i in range(self.num_trials):
            self.plot_trial(i, title + str(i))


    def average_stats(self):
        '''
        return different stats for all trials
        :return: average time, average number of iterations, worst value, best value
        '''
        avg_time = 0
        avg_iterations = 0
        worst_f = -np.inf
        best_f = np.inf

        for i in range(self.num_trials):
            avg_time += self.trial_stats[i]['elapsed_time']
            avg_iterations += self.trial_stats[i]['iterations']
            if self.trial_stats[i]['function_values'][-1] < best_f:
                best_f = self.trial_stats[i]['function_values'][-1]
            if self.trial_stats[i]['function_values'][-1] > worst_f:
                worst_f = self.trial_stats[i]['function_values'][-1]
        avg_time /= self.num_trials
        avg_iterations /= self.num_trials

        return avg_time, avg_iterations, worst_f, best_f

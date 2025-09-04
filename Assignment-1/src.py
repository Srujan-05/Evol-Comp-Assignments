# implementation of all the DE Algorithm related classes.
import numpy as np


class DifferentialEvolution:
    def __init__(self, population_size, no_of_gens, no_design_vars, fitness_func, constraints, crossover_prob, K, optimisation_option):
        self.popl_size = population_size
        self.number_gens = no_of_gens
        self.num_des_vars = no_design_vars
        self.fitness = fitness_func
        self.constraints = constraints
        self.crossover_prob = crossover_prob
        self.K = K
        self.opt_op = optimisation_option

    def mutantVectorGeneration(self):
        pass

    def trialVectorGeneration(self):
        pass

    def run(self):
        pass


class Candidates:
    def __init__(self, num_design_vars):
        self.vector = np.zeros(shape=(num_design_vars, 1))
        self.best = None
        self.best_vec = np.zeros(shape=self.vector.shape)


class Constraints:
    def __init__(self, constraint, type):
        """
        :param constraint: the constraint to be supplied as a function of the design variables with the constraint
                            operation against 0
        :param type: ['<', '>', '<=', '>=', '!='] are the only arguments to be accepted. this
                    tells us what type of constraint we are applying.
        example:
            if the applied constraint on the problem is x < 100, the class initialisation is as:
                constrain = lambda x: x-100 (or) any function that does the similar operation
                type = '<'
            the function then compares it to 0 and sees if the function values are actually valid.
        """
        self.constraint = constraint
        self.type = type


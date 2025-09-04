# implementation of all the DE Algorithm related classes.
import numpy as np


class DifferentialEvolution:
    def __init__(self, population_size, no_of_gens, no_design_vars, fitness_func, constraints, crossover_prob=0.8, K=0.5, optimisation_option="min"):
        """
        :param population_size: The number of candidates in the population
        :param no_of_gens: the number of generations to terminate at if convergence is not achieved
        :param no_design_vars: the number of variables.
        :param fitness_func: the fitness functions
        :param constraints: the list of all constraints
        :param crossover_prob: the crossover probability
        :param K: The K value in the differential evolution
        :param optimisation_option: min or max for minimisation or maximisation respectively
        """

        self.popl_size = population_size
        self.number_gens = no_of_gens
        self.num_des_vars = no_design_vars
        self.fitness = fitness_func
        self.constraints = constraints
        self.crossover_prob = crossover_prob
        self.K = K
        self.opt_op = optimisation_option

    def mutantVectorGeneration(self, candidates, target_cand, F):
        while True:
            vector_r1 = np.random.choice(candidates)
            vector_r2 = np.random.choice(candidates)
            vector_r3 = np.random.choice(candidates)
            if vector_r1.vector != target_cand.vector and vector_r1.vector != vector_r2.vector and vector_r1.vector != vector_r3.vector and vector_r3.vector != vector_r2.vector:
                break

        mutant_vec = target_cand.vector + self.K * (vector_r1.vector - target_cand.vector) + F * (vector_r2.vector - vector_r3.vector)
        mutant_cand = Candidates(vector=mutant_vec)
        return mutant_cand

    def trialVectorGeneration(self, target_cand, mutant_cand):
        trial_vector = target_cand.vector.copy()
        for j in range(len(trial_vector)):
            bit_str_tar = f'{target_cand.vector[j]:0{13}b}'[2:]
            bit_str_mut = f'{mutant_cand.vector[j]:0{13}b}'[2:]
            res_bit = '0b'
            for k in range(len(bit_str_tar)):
                if np.random.random(1) <= self.crossover_prob:
                    res_bit += bit_str_mut[k]
                else:
                    res_bit += bit_str_tar[k]

            trial_vector[j] = int(res_bit, 2)

        trail_cand = Candidates(vector=trial_vector)
        return trail_cand

    def run(self):
        pass


class Candidates:
    def __init__(self, num_design_vars=0, vector=None):
        self.vector = np.zeros(shape=(num_design_vars, )) if vector is None else vector
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


if __name__ == "__main__":
    trial_cand = Candidates(vector=np.array([1, 2, 3, 4]))
    mut_cand = Candidates(vector=np.array([2, 3, 4, 512]))

    de = DifferentialEvolution(0, 1, 4, None, [])

    de.trialVectorGeneration(trial_cand, mut_cand)

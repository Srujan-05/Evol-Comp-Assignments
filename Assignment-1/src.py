# implementation of all the DE Algorithm related classes.
import numpy as np
import random
import struct


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
        self.generations = {}
        self.global_bests = []

    def mutantVectorGeneration(self, candidates, target_cand, F):
        pool = [c for c in candidates if c is not target_cand]
        vector_r1, vector_r2, vector_r3 = random.sample(pool, 3)
        mutant_vec = target_cand.vector + self.K * (vector_r1.vector - target_cand.vector) + F * (vector_r2.vector - vector_r3.vector)
        mutant_cand = Candidates(vector=mutant_vec)
        return mutant_cand

    def trialVectorGeneration(self, target_cand, mutant_cand):
        trial_vector = target_cand.vector.copy()
        for j in range(len(trial_vector)):
            bit_str_tar = self.float_to_bitstr(target_cand.vector[j])
            bit_str_mut = self.float_to_bitstr(mutant_cand.vector[j])
            res_bit = ''
            for k in range(len(bit_str_tar)):
                if np.random.random(1) <= self.crossover_prob:
                    res_bit += bit_str_mut[k]
                else:
                    res_bit += bit_str_tar[k]

            trial_vector[j] = self.bitstr_to_float(res_bit)

        trail_cand = Candidates(vector=trial_vector)
        return trail_cand

    def run(self):
        curr_gen = 0
        current_cands = self.initailiseCandidates()
        while curr_gen < self.number_gens or not self.checkConvergence():
            self.generations[curr_gen] = current_cands
            F = 4 * np.random.random_sample() - 2
            next_cands = []
            for cand in current_cands:
                while True:
                    mutant_cand = self.mutantVectorGeneration(current_cands, cand, F)
                    trail_cand = self.trialVectorGeneration(cand, mutant_cand)
                    valid = True
                    for constraint in self.constraints:
                        if not constraint.checkConstraint(trail_cand):
                            valid = False
                            break
                    if valid:
                        break
                result, _ = self.fitness.checkOptima([trail_cand, cand])
                if result == 0:
                    next_cands.append(trail_cand)
                else:
                    next_cands.append(cand)
            best, vals = self.fitness.checkOptima(next_cands)
            self.global_bests.append(vals[best])
            curr_gen += 1

    def checkConvergence(self):
        return False

    def initailiseCandidates(self):
        candidates = []
        
        for _ in range(self.popl_size):
            while True:
                candidate = Candidates(num_design_vars=self.num_des_vars)

                '''Note: might have to reconsider this range
                 if candidates are to be spread all over the solution space, how would I do that?'''
                candidate.vector = np.random.uniform(-1000,1000, self.num_des_vars)

                valid = True
                for constrain in self.constraints:
                    if not constrain.checkConstraint(candidate):
                        valid = False
                        break

                    if valid:
                        candidates.append(candidate)
                        break
                
        return candidates


    def float_to_bitstr(self, x, bits=64):
        # bits must be 32 or 64
        if bits == 32:
            u = np.array([x], dtype=np.float32).view(np.uint32)[0]
        else:
            u = np.array([x], dtype=np.float64).view(np.uint64)[0]
        return f'{int(u):0{bits}b}'

    def bitstr_to_float(self, bstr, bits=64):
        # ensure correct width
        if len(bstr) < bits:
            bstr = bstr.zfill(bits)
        elif len(bstr) > bits:
            bstr = bstr[-bits:]  # keep the least-significant bits if too long
        i = int(bstr, 2)
        if bits == 32:
            return np.array([i], dtype=np.uint32).view(np.float32)[0]
        else:
            return np.array([i], dtype=np.uint64).view(np.float64)[0]


class Candidates:
    def __init__(self, num_design_vars=0, vector=None):
        self.vector = np.zeros(shape=(num_design_vars, )) if vector is None else vector
        self.best = None
        self.best_vec = np.zeros(shape=self.vector.shape)
       
    def checkConstraints(self, constraints):
        for constraint in constraints:
            if not constraint.checkConstraint(self):
                return False
        return True


class Constraints:
    def __init__(self, constraint, type):
        """
        :param constraint: the constraint to be supplied as a function of the design variables with the constraint
                            operation against 0
        :param type: ['<', '>', '<=', '>=', '!=', '=='] are the only arguments to be accepted. this
                    tells us what type of constraint we are applying.
        example:
            if the applied constraint on the problem is x < 100, the class initialisation is as:
                constrain = lambda x: x-100 (or) any function that does the similar operation
                type = '<'
            the function then compares it to 0 and sees if the function values are actually valid.
        """
        self.constraint = constraint
        self.type = type

    def checkConstraint(self, candidate):
        constraint_value = self.constraint(candidate.vector)
        if self.type == '<':
            return constraint_value < 0
        elif self.type == '>':
            return constraint_value > 0
        elif self.type == '<=':
            return constraint_value <= 0
        elif self.type == '>=':
            return constraint_value >= 0
        elif self.type == '!=':
            return constraint_value != 0
        elif self.type == '==':
            return constraint_value == 0


    """
    Make sure all the trail vectors satisfy constraints -> if not regenerate mutant vector and trail vector
    Not to myself p2 -> where are we comparing fitness of trail vector with parent vector and 
    checking which one to push to the next generation
    how does it integrate with this code?"""
        


if __name__ == "__main__":
    egg_holder_constraints = [
        Constraints(lambda vec: vec[0] - 512, '<='),    # x <= 512
        Constraints(lambda vec: -vec[0] - 512, '<='),   # x >= -512
        Constraints(lambda vec: vec[1] - 512, '<='),    # y <= 512
        Constraints(lambda vec: -vec[1] - 512, '<=')    # y >= -512
    ]

    holder_table_constraints = [
        Constraints(lambda vec: vec[0] - 10, "<="),    # x <= 10
        Constraints(lambda vec: -vec[0] - 10, '<='),   # x >= -10
        Constraints(lambda vec: vec[1] - 10, '<='),    # y <= 10
        Constraints(lambda vec: -vec[1] - 10, '<=')    # y >= -10
    ]


    trial_cand = Candidates(vector=np.array([1, 2, 3, 4]))
    mut_cand = Candidates(vector=np.array([2, 3, 4, 512]))

    de = DifferentialEvolution(0, 1, 4, None, [])

    de.trialVectorGeneration(trial_cand, mut_cand)

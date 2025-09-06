# implementation of all the DE Algorithm related classes.
import numpy as np
import random
from utils import FitnessFunc


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
        self.convergence_threshold = 1e-9
        self.stagnation_count = 0
        self.max_stagnation = 50

    def mutantVectorGeneration(self, candidates, target_cand, F):
        pool = [c for c in candidates if c is not target_cand]
        vector_r1, vector_r2, vector_r3 = random.sample(pool, 3)
        mutant_vec = target_cand.vector + self.K * (vector_r1.vector - target_cand.vector) + F * (vector_r2.vector - vector_r3.vector)
        mutant_vec = np.clip(mutant_vec, -512, 512)
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

        trial_vector = np.clip(trial_vector, -512, 512)
        trail_cand = Candidates(vector=trial_vector)
        return trail_cand

    def run(self):
        curr_gen = 0
        current_cands = self.initailiseCandidates()
        while curr_gen < self.number_gens and not self.checkConvergence():
            self.generations[curr_gen] = current_cands
            F = np.random.uniform(-2, 2)
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
            self.global_bests.append((vals[best], next_cands[best]))
            curr_gen += 1
            current_cands = next_cands

    def checkConvergence(self):
        if len(self.global_bests) < 2:
            return False

        recent_improvement = abs(self.global_bests[-1][0] - self.global_bests[-2][0])
        if recent_improvement < self.convergence_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        return self.stagnation_count >= self.max_stagnation

    def calculatePopulationDiversity(self, population):
        """Calculate the diversity of the population"""
        if len(population) <= 1:
            return 0
        
        vectors = np.array([cand.vector for cand in population])
        mean_vector = np.mean(vectors, axis=0)
        diversity = np.mean(np.sqrt(np.sum((vectors - mean_vector) ** 2, axis=1)))
        return diversity

    def initailiseCandidates(self):
        candidates = []
        
        for _ in range(self.popl_size):
            while True:
                candidate = Candidates(num_design_vars=self.num_des_vars)

                '''Note: might have to reconsider this range
                 if candidates are to be spread all over the solution space, how would I do that?'''
                # candidate.vector = np.random.uniform(-512, 512, self.num_des_vars)
                scale = 512
                candidate.vector = np.random.uniform(-0.5, 0.5, self.num_des_vars) * scale * 2

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


if __name__ == "__main__":
    egg_holder_constraints = [
        Constraints(lambda vec: vec[0] - 512, '<='),    # x <= 512
        Constraints(lambda vec: vec[0] + 512, '>='),   # x >= -512
        Constraints(lambda vec: vec[1] - 512, '<='),    # y <= 512
        Constraints(lambda vec: vec[1] + 512, '>=')    # y >= -512
    ]

    def eggholder(v):
        x, y = v[0], v[1]
        return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))
    egg_holder_function = FitnessFunc(eggholder)

    diffEvol = DifferentialEvolution(20, 50, 2, egg_holder_function, egg_holder_constraints, K=0.5)
    diffEvol.run()

    sol = diffEvol.global_bests[-1]
    print(sol[0], sol[1].vector)

# implementation and usage of the DE to EggHolder and HölderTable problems
from src import *
from utils import *

if __name__ == "__main__":
    populations = [20, 50, 100, 200]
    generations = [50, 200]


    def eggholder(v):
        x, y = v[0], v[1]
        return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))


    egg_holder_function = FitnessFunc(eggholder)

    constraint1 = Constraints(lambda x: x[0] - 512, '<=')
    constraint2 = Constraints(lambda x: x[1] - 512, '<=')
    constraint3 = Constraints(lambda x: x[0] + 512, '>=')
    constraint4 = Constraints(lambda x: x[1] + 512, '>=')

    constraints = [constraint1, constraint2, constraint3, constraint4]

    # def holderTable(v):
    #     x, y = v[0], v[1]
    #     x, y = v[0], v[1]
    #     return -abs(np.sin(x) * np.cos(y) * np.exp(abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi)))
    #
    #
    # holder_function = FitnessFunc(holderTable)
    #
    # constraint1 = Constraints(lambda x: x[0] - 10, '<=')
    # constraint2 = Constraints(lambda x: x[1] - 10, '<=')
    # constraint3 = Constraints(lambda x: x[0] + 10, '>=')
    # constraint4 = Constraints(lambda x: x[1] + 10, '>=')
    #
    # constraints = [constraint1, constraint2, constraint3, constraint4]

    for generation in generations:
        for population in populations:
            diffEvol = DifferentialEvolution(population, generation, 2, holder_function, constraints)
            diffEvol.run()
            sol = diffEvol.global_bests[-1]
            print("Population size: {}, Generations: {}, Value: {}, Solution Vector: {}, ".format(population, generation, sol[0], sol[1].vector))
            Visualization().visualizeBestConvergence(diffEvol)

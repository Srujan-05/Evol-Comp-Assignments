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

    constraint1 = Constraints(lambda x: x[0] - 512, '<')
    constraint2 = Constraints(lambda x: x[1] - 512, '<')
    constraint3 = Constraints(lambda x: x[0] + 512, '>')
    constraint4 = Constraints(lambda x: x[1] + 512, '>')

    constraints = [constraint1, constraint2, constraint3, constraint4]

    for population in populations:
        for generation in generations:
            diffEvol = DifferentialEvolution(population, generation, 2, egg_holder_function, constraints)
            diffEvol.run()
            sol = diffEvol.global_bests[-1]

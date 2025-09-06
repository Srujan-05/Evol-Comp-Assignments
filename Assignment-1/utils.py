# implementation of the results visualization class.
import numpy as np
import matplotlib.pyplot as plt


class FitnessFunc:
    def __init__(self, fitness_function, optimisation_option: str = "min", known_optimum=None):
        """
        fitness_function: callable(vec: np.ndarray) -> float
        optimisation_option: "min" or "max"
        known_optimum: optional number used only for reporting if you want
        """
        self.fit_func = fitness_function
        self.opt = optimisation_option.lower()
        self.known_optimum = known_optimum

    def _eval_vec(self, vec):
        vec = np.asarray(vec, dtype=float)
        return float(self.fit_func(vec))

    def checkOptima(self, candidates):
        """
        Accepts a list of Candidate objects (with .vector) OR raw vectors.
        Returns (best_index, list_of_fitness_values)
        """
        vals = []
        for item in candidates:
            if hasattr(item, "vector"):
                vals.append(self._eval_vec(item.vector))
            else:
                vals.append(self._eval_vec(item))
        if self.opt == "max":
            best_idx = int(np.argmax(vals))
        else:
            best_idx = int(np.argmin(vals))
        return best_idx, vals


class Visualization:
    """
    Plot utilities:

    - visualizeConvergenceHistory(de, ...):
        line plot of best and mean population fitness across generations.

    - visualizeBestSolutionHistory(de, ...):
        if no. of design variables == 2 -> (x, y) trajectory in design space;
        else -> each component of best vector vs generation.
    """

    @staticmethod
    def _best_and_avg_per_generation(de):
        gens_sorted = sorted(de.generations.keys())
        best_hist, avg_hist = [], []

        for g in gens_sorted:
            pop = de.generations[g]
            _, vals = de.fitness.checkOptima(pop)
            vals = np.asarray(vals, dtype=float)
            best = np.max(vals) if de.fitness.opt == "max" else np.min(vals)
            best_hist.append(best)
            avg_hist.append(float(np.mean(vals)))

        return np.asarray(best_hist), np.asarray(avg_hist)

    @staticmethod
    def visualizeConvergenceHistory(
        de,
        title = "Convergence (DE)",
        use_symlog = False,
        savepath = None,
        show = True,
    ):
        if not de.generations:
            raise ValueError("No generations recorded. Run de.run() before plotting.")

        best_hist, avg_hist = Visualization._best_and_avg_per_generation(de)
        gens = np.arange(len(best_hist))

        plt.figure(figsize=(8, 5))
        plt.plot(gens, best_hist, label="Best fitness")
        plt.plot(gens, avg_hist, label="Average fitness")
        if use_symlog:
            plt.yscale("symlog")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def visualizeBestSolutionHistory(
        de,
        title = None,
        savepath = None,
        show = True,
    ):
        if not de.generations:
            raise ValueError("No generations recorded. Run de.run() before plotting.")

        gens_sorted = sorted(de.generations.keys())
        best_vecs = []

        for g in gens_sorted:
            pop = de.generations[g]
            best_idx, _ = de.fitness.checkOptima(pop)
            best_vecs.append(np.asarray(pop[best_idx].vector, dtype=float))

        best_mat = np.vstack(best_vecs)  # shape: (num_gens, dim)
        dim = best_mat.shape[1]

        if dim == 2:
            # plot path in (x, y) space
            plt.figure(figsize=(6, 6))
            plt.plot(best_mat[:, 0], best_mat[:, 1], marker="o")
            # draw arrows to show direction
            for i in range(len(best_mat) - 1):
                x0, y0 = best_mat[i]
                x1, y1 = best_mat[i + 1]
                plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                             arrowprops=dict(arrowstyle="->", lw=1, alpha=0.7))
            plt.xlabel("x₁")
            plt.ylabel("x₂")
            plt.title(title or "Best-solution trajectory (design space)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        else:
            # components vs generation
            plt.figure(figsize=(8, 5))
            gens = np.arange(best_mat.shape[0])
            for d in range(dim):
                plt.plot(gens, best_mat[:, d], label=f"x[{d}]")
            plt.xlabel("Generation")
            plt.ylabel("Component value")
            plt.title(title or "Best-solution components vs generation")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()


import numpy as np
from typing import Union

class DE(object):
    def __init__(self, 
        pop_size : Union[int,None] = None,
        crossover_prob : float = 0.9,
        mutation_factor : float = 0.8) -> None:

        assert 0 <= mutation_factor <= 2, ValueError("mutation_factor not in range [0, 2]") 
        
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor

    def init_population(self, bounds: np.ndarray, dim: int) -> np.ndarray :
        # Initialize population from standard uniform distribution and scale to bounds 
        population = bounds[:, 0] + (np.random.rand(self.pop_size, dim) * (bounds[:, 1] - bounds[:, 0]))
        return population

    def eval_population(self, obj, population: np.ndarray, budget) -> np.ndarray:
        fitness = np.zeros(self.pop_size)
        for idx, candidate in enumerate(population):
            fitness[idx] = obj(candidate)
        return fitness

    def mutation(self, population: np.ndarray):
        # TODO : not purely random. sampled configurations must be distinct
        # from candidate
        selection = np.random.choice(np.arange(self.pop_size), 3, replace=False)
        base, a, b = population[selection]

        diff = a - b
        mutant = base + self.mutation_factor * diff
        return mutant

    def crossover(self, candidate: np.ndarray, mutant: np.ndarray, dim: int) -> np.ndarray:
        # Sample a random value from U[0, 1] for every dimension
        p = np.random.rand(dim)
        # perform binomial crossover
        child = [mutant[i] if p[i] < self.crossover_prob else candidate[i] for i in range(dim)]
        return np.asarray(child)

    def selection(self, obj, children: list, population: np.ndarray, fitness: np.ndarray, budget):
        # Conduct parent-child competition and select new population 
        children_fitness = self.eval_population(obj, children, budget)
        for i in range(self.pop_size):
            if children_fitness[i] <= fitness[i]:
                population[i] = children[i]
                fitness[i] = children_fitness[i]
        
        return population, fitness

    def check_bounds(self, mutant, bounds, dim: int) -> np.ndarray:
        # Clip values to range defined in bounds
        mutant_clipped = [np.clip(mutant[i], bounds[i, 0], bounds[i, 1]) for i in range(dim)]
        return mutant_clipped

    def optimize(self, obj, bounds, budget=None, iter=10):
        
        # Dimensionality of a hyperparameter configuration
        dim = len(bounds)

        if self.pop_size is None:
            self.pop_size = 10 * dim # heuristic

        # Initialize and Evaluate the population
        population = self.init_population(bounds, dim)
        fitness = self.eval_population(obj, population, budget)

        # Until budget is exhausted
        for t in range(iter):
            children = []
            for candidate in population:
                mutant = self.mutation(population)
                mutant = self.check_bounds(mutant, bounds, dim)

                child = self.crossover(candidate, mutant, dim)
                children.append(child)

            population, fitness = self.selection(obj, children, population, fitness, budget)
    
        best_candidate = np.argmin(fitness)
        return population[best_candidate]


if __name__ == "__main__":
    de = DE(10)
    def obj(x):
        return np.linalg.norm(x)

    bounds = np.asarray([(-5.0, 5.0), (-5.0, 5.0)])
    print(de.optimize(obj, bounds, iter=10))




        


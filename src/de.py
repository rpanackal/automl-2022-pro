from typing import Callable, Union

import ConfigSpace
import numpy as np
from ConfigSpace.util import (deactivate_inactive_hyperparameters,
                              impute_inactive_values)

SEED = 42


class ConfigVectorSpace(ConfigSpace.ConfigurationSpace):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hyperparameters = self.get_hyperparameters()
        self.dim = len(self.hyperparameters)

        self.name_to_idx = {}
        for i, hp in enumerate(self.hyperparameters):
            # maps hyperparameter name to positional index in vector form
            self.name_to_idx[hp.name] = i
        
    def sample_vectors(self, size):
        configurations = super().sample_configuration(size)
        vectors = [self._to_vector(config) for config in configurations]
        return vectors
    
    def _to_vector(self, config: ConfigSpace.Configuration) -> np.array:
        '''Converts ConfigSpace.Configuration object to numpy array scaled to [0,1]
        Works when self is a ConfigVectorSpace object and the input config is a ConfigSpace.Configuration object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        '''
        
        config = impute_inactive_values(config)

        #TODO: getrid of nan
        vector = [np.nan for i in range(self.dim)]
        self.name_to_id = dict()

        for name in config:
            idx = self.name_to_idx[name]
            hyper = self[name]
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                nlevels = len(hyper.sequence)
                vector[idx] = hyper.sequence.index(config[name]) / nlevels
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                nlevels = len(hyper.choices)
                vector[idx] = hyper.choices.index(config[name]) / nlevels
            else:
                bounds = (hyper.lower, hyper.upper)
                param_value = config[name]
                if hyper.log:
                    vector[idx] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                else:
                    vector[idx] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
        return np.array(vector)
    
    def to_config(self, vector: np.array) -> ConfigSpace.Configuration:
        '''Converts numpy array to ConfigSpace.Configuration object
        Works when self is a ConfigVectorSpace object and the input vector is in the domain [0, 1].
        '''
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = impute_inactive_values(self.sample_configuration()).get_dictionary()

        # iterates over all hyperparameters and normalizes each based on its type
        for hyper in self.hyperparameters:
            idx = self.name_to_idx[hyper.name]

            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[idx] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
                param_value = hyper.choices[np.where((vector[idx] < ranges) == False)[0][-1]]
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[idx] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[idx]
                if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                    param_value = int(np.round(param_value))  # converting to discrete (int)
                else:
                    param_value = float(param_value)
            new_config[hyper.name] = param_value
        # the mapping from unit hypercube to the actual config space may lead to illegal
        # configurations based on conditions defined, which need to be deactivated/removed
        new_config = deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self
        )
        return new_config



class DE(object):
    def __init__(self,
        space : ConfigVectorSpace,
        pop_size : Union[int,None] = None,
        crossover_prob : float = 0.9,
        mutation_factor : float = 0.8,
        bound_control = "random",
        rs: np.random.RandomState=None) -> None:

        assert 0 <= mutation_factor <= 2, ValueError("mutation_factor not in range [0, 2]") 
        
        self.space = space
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor
        self.bound_control = bound_control
        self.rs = rs

        if self.pop_size is None:
            self.pop_size = 10 * self.space.dim # heuristic

    def init_population(self) -> np.ndarray :

        # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
        population = self.space.sample_vectors(size=self.pop_size)
        return np.asarray(population)
    
    def eval_population(self, obj : Callable, population: Union[np.ndarray, list] , budget) -> np.ndarray:
        fitness = np.zeros(self.pop_size)
        for idx, candidate in enumerate(population):
            candidate = self.space.to_config(candidate)
            fitness[idx] = obj(candidate, budget)
        return fitness
    
    def mutation(self, population: np.ndarray):
        # TODO : not necessarily purely random. sampled configurations can be distinct
        # from candidate
        selection = self.rs.choice(np.arange(self.pop_size), 3, replace=False)
        base, a, b = population[selection]

        diff = a - b
        mutant = base + self.mutation_factor * diff
        return mutant

    def crossover(self, candidate: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        # Sample a random value from U[0, 1] for every dimension
        p = self.rs.rand(self.space.dim)

        # perform binomial crossover
        child = np.asarray([mutant[i] if p[i] < self.crossover_prob else candidate[i] for i in range(self.space.dim)])
        return child
    
    def selection(self, obj, children: list, population: np.ndarray, fitness: np.ndarray, budget):
        # Conduct parent-child competition and select new population 
        children_fitness = self.eval_population(obj, children, budget)
        for i in range(self.pop_size):
            if children_fitness[i] <= fitness[i]:
                population[i] = children[i]
                fitness[i] = children_fitness[i]
        
        return population, fitness
    
    def check_bounds(self, vector: np.ndarray) -> np.ndarray:

        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.bound_control == 'random':
            vector[violations] = self.rs.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            # Can be exploited by optimizer if solution at clip limits 
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector


    def optimize(self, obj : Callable, budget=None, iter: int = 10):
        # Initialize and Evaluate the population
        population = self.init_population()
        fitness = self.eval_population(obj, population, budget)

        # Until budget is exhausted
        for t in range(iter):
            children = []
            for candidate in population:
                mutant = self.mutation(population)
                mutant = self.check_bounds(mutant)

                child = self.crossover(candidate, mutant)
                children.append(child)

            population, fitness = self.selection(obj, children, population, fitness, budget)
        
        best_candidate = np.argmin(fitness)
        return self.space.to_config(population[best_candidate])

def obj(x, budget):
    """Sample objective function"""
    y = list()
    for name in x:
        if type(x[name]) is str:
            y.append(len(x[name]))
        else:
            y.append(x[name])

    results = np.linalg.norm(y) 
    return results

if __name__ == "__main__":

    rs = np.random.RandomState(seed=SEED)
    space = ConfigVectorSpace(
        name="neuralnetwork",
        seed=SEED,
        space={
            "lr": ConfigSpace.UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-1, log=True, default_value=1e-3),
            "dropout": ConfigSpace.UniformFloatHyperparameter("dropout", lower=0, upper=0.5, default_value=0.3),
            #"reg_const": ConfigSpace.Float("lambda",),
            "reg_type": ConfigSpace.CategoricalHyperparameter("reg_type", choices=["l1", "l2"], weights=[0.5, 0.5], default_value="l2"),
            "depth": ConfigSpace.Integer("depth", bounds=[2, 9]),
            "batch_size": ConfigSpace.OrdinalHyperparameter("batch_size", sequence=[16, 32, 64, 128], default_value=16)
        },
    )

    de = DE(space, 10, rs=rs)
    print("Best configuration", de.optimize(obj, space, iter=100))

from typing import List, Union
from unittest import result

import ConfigSpace
from ConfigSpace.util import impute_inactive_values, deactivate_inactive_hyperparameters

import numpy as np
import constants


class DE(object):
    def __init__(self, 
        pop_size : Union[int,None] = None,
        crossover_prob : float = 0.9,
        mutation_factor : float = 0.8,
        bound_control = "random") -> None:

        assert 0 <= mutation_factor <= 2, ValueError("mutation_factor not in range [0, 2]") 
        
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor
        self.bound_control = bound_control 

    def init_population(self, space: Union[np.ndarray, ConfigSpace.ConfigurationSpace], dim: int) -> np.ndarray :

        # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
        population = space.sample_configuration(size=self.pop_size)
        if not isinstance(population, List):
            population = [population]
        # the population is maintained in a list-of-vector form where each ConfigSpace
        # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]

        print("Candidate as ConfigSpace condfigurations  internal array")
        for candidate in population:
            print(candidate.get_array())

        population = [self.configspace_to_vector(candidate) for candidate in population]

        print("Candidate as ConfigSpace configspace_to_vector result")
        for candidate in population:
            print(candidate)
        exit(0)
        return np.asarray(population)

    def eval_population(self, obj, population: np.ndarray, budget) -> np.ndarray:
        fitness = np.zeros(self.pop_size)
        for idx, candidate in enumerate(population):
            candidate = self.vector_to_configspace(candidate)
            fitness[idx] = obj(candidate)
        return fitness

    def mutation(self, population: np.ndarray):
        # TODO : not necessarily purely random. sampled configurations can be distinct
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
        child = np.asarray([mutant[i] if p[i] < self.crossover_prob else candidate[i] for i in range(dim)])
        return child

    def selection(self, obj, children: list, population: np.ndarray, fitness: np.ndarray, budget):
        # Conduct parent-child competition and select new population 
        children_fitness = self.eval_population(obj, children, budget)
        for i in range(self.pop_size):
            if children_fitness[i] <= fitness[i]:
                population[i] = children[i]
                fitness[i] = children_fitness[i]
        
        return population, fitness

    def check_bounds(self, mutant, space, dim: int) -> np.ndarray:
        vector = mutant
        # # Clip values to range defined in bounds

        # if self.bound_control == "random":
        #     mutant = np.random.rand(dim) 
        # else:
        #     # Can be exploited by optimizer if solution at clip limits 
        #     mutant = [np.clip(mutant[i], 0, 1) for i in range(dim)]
        # return np.asarray(mutant)
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.bound_control == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def optimize(self, obj, space, budget=None, iter=10):
        self.space = space

        self.hps = {}
        for i, hp in enumerate(space.get_hyperparameters()):
            # maps hyperparameter name to positional index in vector form
            self.hps[hp.name] = i
        
        # Dimensionality of a hyperparameter configuration
        dim = len(space.get_hyperparameters())

        if self.pop_size is None:
            self.pop_size = 10 * dim # heuristic

        # Initialize and Evaluate the population
        population = self.init_population(space, dim)
        fitness = self.eval_population(obj, population, budget)

        # Until budget is exhausted
        for t in range(iter):
            children = []
            for candidate in population:
                mutant = self.mutation(population)
                mutant = self.check_bounds(mutant, space, dim)

                child = self.crossover(candidate, mutant, dim)
                children.append(child)

            population, fitness = self.selection(obj, children, population, fitness, budget)
        
        best_candidate = np.argmin(fitness)
        return self.vector_to_configspace(population[best_candidate])

    def vector_to_configspace(self, vector: np.array) -> ConfigSpace.Configuration:
        '''Converts numpy array to ConfigSpace object
        Works when self.space is a ConfigSpace object and the input vector is in the domain [0, 1].
        '''
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = impute_inactive_values(
            self.space.sample_configuration()
        ).get_dictionary()
        # iterates over all hyperparameters and normalizes each based on its type
        for i, hyper in enumerate(self.space.get_hyperparameters()):
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
                param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                    param_value = int(np.round(param_value))  # converting to discrete (int)
                else:
                    param_value = float(param_value)
            new_config[hyper.name] = param_value
        # the mapping from unit hypercube to the actual config space may lead to illegal
        # configurations based on conditions defined, which need to be deactivated/removed
        new_config = deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self.space
        )
        return new_config

    def configspace_to_vector(self, config: ConfigSpace.Configuration) -> np.array:
        '''Converts ConfigSpace object to numpy array scaled to [0,1]
        Works when self.space is a ConfigSpace object and the input config is a ConfigSpace object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        '''
        # the imputation replaces illegal parameter values with their default
        config = impute_inactive_values(config)
        dimensions = len(self.space.get_hyperparameters())
        vector = [np.nan for i in range(dimensions)]
        for name in config:
            i = self.hps[name]
            hyper = self.space.get_hyperparameter(name)
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                nlevels = len(hyper.sequence)
                vector[i] = hyper.sequence.index(config[name]) / nlevels
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                nlevels = len(hyper.choices)
                vector[i] = hyper.choices.index(config[name]) / nlevels
            else:
                bounds = (hyper.lower, hyper.upper)
                param_value = config[name]
                if hyper.log:
                    vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                else:
                    vector[i] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
        return np.array(vector)


def obj(x):
    y = list()
    for name in x:
        if type(x[name]) is str:
            y.append(len(x[name]))
        else:
            y.append(x[name])

    x = y
    results = np.linalg.norm(x) 
    return results




class ConfigurationVector(np.ndarray):
    def __new__(cls, config : ConfigSpace.Configuration):

        config, space, dimensions = cls.get_properties(config)

        obj = np.asarray(cls.configspace_to_vector(config)).view(cls)

        obj.config = config
        obj.space  = space
        obj.dimensions = dimensions
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        self.config = getattr(obj, 'config', None)
        self.space = getattr(obj, 'space', None)
        self.dimensions = getattr(obj, 'dimensions', None)
    
    @staticmethod
    def configspace_to_vector(config: ConfigSpace.Configuration) -> np.array:
        '''Converts ConfigSpace object to numpy array scaled to [0,1]
        Works when self.space is a ConfigSpace object and the input config is a ConfigSpace object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        '''
        
        config, space, dimensions = ConfigurationVector.get_properties(config)

        vector = [np.nan for i in range(dimensions)]
        name_to_id = dict()

        for i, name in enumerate(config):
            name_to_id[name] = i
            hyper = space.get_hyperparameter(name)
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                nlevels = len(hyper.sequence)
                vector[i] = hyper.sequence.index(config[name]) / nlevels
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                nlevels = len(hyper.choices)
                vector[i] = hyper.choices.index(config[name]) / nlevels
            else:
                bounds = (hyper.lower, hyper.upper)
                param_value = config[name]
                if hyper.log:
                    vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                else:
                    vector[i] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
        return np.array(vector)

    @staticmethod
    def get_properties(config):
        # the imputation replaces illegal parameter values with their default
        config = impute_inactive_values(config)
        space = config.configuration_space
        dimensions = len(space.get_hyperparameters())

        return config, space, dimensions

# class CustomConfigurationSpace(ConfigSpace.ConfigurationSpace):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
    
#     def config_to_vector(self, vector: np.array) -> ConfigSpace.Configuration:
#         '''Converts numpy array to ConfigSpace object
#         Works when self.space is a ConfigSpace object and the input vector is in the domain [0, 1].
#         '''
#         # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
#         new_config = impute_inactive_values(
#             self.space.sample_configuration()
#         ).get_dictionary()
#         # iterates over all hyperparameters and normalizes each based on its type
#         for i, hyper in enumerate(self.space.get_hyperparameters()):
#             if type(hyper) == ConfigSpace.OrdinalHyperparameter:
#                 ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
#                 param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
#             elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
#                 ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
#                 param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
#             else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
#                 # rescaling continuous values
#                 if hyper.log:
#                     log_range = np.log(hyper.upper) - np.log(hyper.lower)
#                     param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
#                 else:
#                     param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
#                 if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
#                     param_value = int(np.round(param_value))  # converting to discrete (int)
#                 else:
#                     param_value = float(param_value)
#             new_config[hyper.name] = param_value
#         # the mapping from unit hypercube to the actual config space may lead to illegal
#         # configurations based on conditions defined, which need to be deactivated/removed
#         new_config = deactivate_inactive_hyperparameters(
#             configuration = new_config, configuration_space=self.space
#         )
#         return new_config

#     def configspace_to_vector(self, config: ConfigSpace.Configuration) -> np.array:
#         '''Converts ConfigSpace object to numpy array scaled to [0,1]
#         Works when self.space is a ConfigSpace object and the input config is a ConfigSpace object.
#         Handles conditional spaces implicitly by replacing illegal parameters with default values
#         to maintain the dimensionality of the vector.
#         '''
#         # the imputation replaces illegal parameter values with their default
#         config = impute_inactive_values(config)
#         dimensions = len(self.space.get_hyperparameters())
#         vector = [np.nan for i in range(dimensions)]
#         for name in config:
#             i = self.hps[name]
#             hyper = self.space.get_hyperparameter(name)
#             if type(hyper) == ConfigSpace.OrdinalHyperparameter:
#                 nlevels = len(hyper.sequence)
#                 vector[i] = hyper.sequence.index(config[name]) / nlevels
#             elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
#                 nlevels = len(hyper.choices)
#                 vector[i] = hyper.choices.index(config[name]) / nlevels
#             else:
#                 bounds = (hyper.lower, hyper.upper)
#                 param_value = config[name]
#                 if hyper.log:
#                     vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
#                 else:
#                     vector[i] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
#         return np.array(vector)


if __name__ == "__main__":
    # TODO: Manage seed for randomstate of numpy and ConfigSpace

    de = DE(10)
    
    space = ConfigSpace.ConfigurationSpace(
        name="neuralnetwork",
        seed=constants.SEED,
        # space={
        #     "a": ConfigSpace.Float("a", bounds=(0.1, 1.5), distribution=ConfigSpace.Normal(1, 10), log=True),
        #     "b": ConfigSpace.Integer("b", bounds=(2, 10)),
        #     "c": ConfigSpace.Categorical("c", ["mouse", "cat", "dog"], weights=[2, 1, 1]),
        # },
        space={
            "a": ConfigSpace.UniformFloatHyperparameter("a", lower=-5.0, upper=5.0),
            "b": ConfigSpace.UniformFloatHyperparameter("b", lower=-5.0, upper=5.0),
            #"c": ConfigSpace.Categorical("c", ["mouse", "cat", "elephant"], weights=[2, 1, 1])
        },
    )

    #print("Best configuration", de.optimize(obj, space, iter=100))

    sample_config = space.sample_configuration(1)
    print("Sample ConfigSpace Configuration", sample_config)

    sample_config_vector = ConfigurationVector(sample_config)
    print("Sample Configuration Vector",sample_config_vector.space)

    #a = np.array([10, 20])
    a = sample_config_vector[:1]
    print("a Vector",a.space)




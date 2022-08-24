import json
import os
import time
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
        if not isinstance(configurations , list):
            configurations = [configurations]

        vectors = [self._to_vector(config) for config in configurations]
        return vectors
    
    def _to_vector(self, config: ConfigSpace.Configuration) -> np.array:
        '''Converts ConfigSpace.Configuration object to numpy array scaled to [0,1]
        Works when self is a ConfigVectorSpace object and the input config is a ConfigSpace.Configuration object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        '''        
        config = impute_inactive_values(config)

        vector = [None] * self.dim
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

        #self.check_configuration(ConfigSpace.Configuration(self, values=new_config))
        #new_config = impute_inactive_values(ConfigSpace.Configuration(self, values=new_config))
        # the mapping from unit hypercube to the actual config space may lead to illegal
        # configurations based on conditions defined, which need to be deactivated/removed
        new_config = deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self
        )
        return new_config


class DE(object):
    def __init__(self,
        space : ConfigVectorSpace,
        crossover_prob : float = 0.9,
        mutation_factor : float = 0.8,
        metric = "f1_Score",
        mode = "max",
        rs: np.random.RandomState=None,
        bound_control = "random",
        save_path : Union[str, None] =".",
        save_freq=10) -> None:

        assert 0 <= crossover_prob <= 1, ValueError("crossover_prob given is not a probability")
        assert 0 <= mutation_factor <= 2, ValueError("mutation_factor not in range [0, 2]")
        assert mode in ["min", "max"], ValueError("Valid optimization mode in ['min', 'max']")
        
        self.space = space
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor
        self.metric = metric
        self.mode = mode
        self.rs = rs
        self.bound_control = bound_control
        self.save_path = save_path
        self.save_freq = save_freq
        
        self.traj = []
        self.runtime = []
        self.inc_config = None
        self.inc_score = float("inf") if self.mode == "min" else float("-inf")

        self.histroy = []

        self._min_pop_size = 3
        self._eval_counter = -1
        self._iteration_counter = -1

    def _init_population(self, pop_size : int) -> np.ndarray :

        # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
        population = self.space.sample_vectors(size=pop_size)
        return np.asarray(population)
    
    def _eval_population(self, obj : Callable, population: Union[np.ndarray, list] , budget, **kwargs) -> np.ndarray:
        fitness = []

        for candidate in population:
            config = self.space.to_config(candidate)
            result = obj(config, budget, **kwargs)

            assert isinstance(result, dict), TypeError("Objective function must return a dictionary")
            assert self.metric in result, KeyError(f"Given mteric '{self.metric}' not found in dictionary {result}, returned by the objective function")
            score = result[self.metric]

            condition = {
                "min" : score < self.inc_score,
                "max" : score > self.inc_score, 
            } 

            if condition[self.mode]:
                self.inc_config = config
                self.inc_score = score

            fitness.append(score)
            # trajectory updated at every fn eval, regardless of save_freq
            self.traj.append(self.inc_score)
            self.runtime.append(self._time_elapsed())

            self._eval_counter += 1

            # Update history and wrtie data every 'save_freq' objective fn evaluations
            if self._eval_counter % self.save_freq == 0:
                self._update_history(candidate, result, budget)
                self.save_data()

        return np.asarray(fitness)

    def _update_history(self, candidate, result, budget):
        record = {
            "candidate": candidate.tolist(),
            "budget": budget}
        
        record.update(result)
        self.histroy.append(record)

    def _sample(self, population, size, replace=False):
        selection = self.rs.choice(np.arange(len(population)), size, replace=replace)
        return population[selection]

    def _mutation(self, population: np.ndarray):
        pop_size = len(population)
        assert pop_size > self._min_pop_size, ValueError(f"Population too small ( < DE()._min_pop_size = {self._min_pop_size}) for mutation")

        base, a, b = self._sample(population, self._min_pop_size)

        diff = a - b
        mutant = base + self.mutation_factor * diff
        return mutant

    def _crossover(self, candidate: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        # Sample a random value from U[0, 1] for every dimension
        p = self.rs.rand(self.space.dim)

        # perform binomial crossover
        child = np.asarray([mutant[i] if p[i] < self.crossover_prob else candidate[i] for i in range(self.space.dim)])
        return child
    
    def _selection(self, population: np.ndarray, children: np.ndarray, fitness: np.ndarray, children_fitness: np.ndarray):
        # Conduct parent-child competition and select new population 
        pop_size = len(population)
        for i in range(pop_size):
            condition = {
                "max" : children_fitness[i] >= fitness[i],
                "min" : children_fitness[i] <= fitness[i]
                }
            if condition[self.mode]:
                population[i] = children[i]
                fitness[i] = children_fitness[i]
        
        return population, fitness
    
    def _check_bounds(self, vector: np.ndarray) -> np.ndarray:

        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.bound_control == 'random':
            vector[violations] = self.rs.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            # Can be exploited by optimizer if solution at clip limits 
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def _next_generation(self, population, alt_pop=None):
        children = []

        for candidate in population:
            # If alt_pop is None, vanilla mutation is perfomed
            mutant = self._mutation(alt_pop if alt_pop is not None else population)
            mutant = self._check_bounds(mutant)
            child = self._crossover(candidate, mutant)
            children.append(child)
        
        return children

    def optimize(self, obj : Callable, budget=None, pop_size : Union[int,None] = None, limit: int = 10, unit : str = "iter", **kwargs):
        self._start_timer()

        if pop_size is None:
            pop_size = 10 * self.space.dim # heuristic

        # Initialize and Evaluate the population
        population = self._init_population(pop_size)
        fitness = self._eval_population(obj, population, budget, **kwargs)

        # Until budget is exhausted
        while not self._is_termination(limit, unit):
            self._iteration_counter += 1

            children = self._next_generation(population)

            children_fitness = self._eval_population(obj, children, budget)
            population, fitness = self._selection(population, children, fitness, children_fitness)
        
        return self.inc_config
    
    def _start_timer(self):
        self._wall_clock_start = time.time()
    
    def _time_elapsed(self):
        # time difference in sec
        diff = time.time() - self._wall_clock_start
        return diff

    def _is_termination(self, limit : int, unit : str):
        assert unit in ["hr", "min", "sec", "evals", "iter"], ValueError("Unrecognized unit given")

        if unit in ["hr", "min", "sec"]:
            scale = {
                "sec" : 1,
                "min" : 60,
                "hr"  : 60 * 60
            }
            diff = self._time_elapsed()
            return diff >= limit * scale[unit]
        elif unit == "evals":
            return self._eval_counter + 1 > limit
        else:
            return self._iteration_counter + 1 > limit

    def save_data(self):
        data = {
            "params" : self._init_params(),
            "result" : {
                "best_config": self.inc_config.get_dictionary(),
                "best_score" : self.inc_score,
            },
            "traj": self.traj,
            "runtime": self.runtime,
            "history" : self.histroy,
        }

        if self.save_path is not None:
            with open(os.path.join(self.save_path, "data.json"), "w") as outfile:
                json.dump(data, outfile)
    
    def _init_params(self):
        params = {
            "crossover_prob" : self.crossover_prob,
            "mutation_factor" : self.mutation_factor,
            "metric" : self.metric,
            "mode" : self.mode,
            "seed" : SEED,
            "bound_control" : self.bound_control,
            "iters" : self._iteration_counter,
            "evals" : self._eval_counter

        }
        return params

class DEHB(DE):
    def __init__(self, 
        space: ConfigVectorSpace,
        min_budget : int = 10,
        max_budget : int = 100,
        eta : int = 2,
        crossover_prob: float = 0.9, 
        mutation_factor: float = 0.8, 
        metric = "f1_score",
        mode = "max",
        rs: np.random.RandomState=None,
        bound_control = "random") -> None:

        super().__init__(space=space,
            crossover_prob=crossover_prob, 
            mutation_factor=mutation_factor,
            metric=metric,
            mode=mode,
            bound_control=bound_control, 
            rs=rs)
        
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.eta = eta
        
        self._all_in_one = self._get_bracket()
        self._SH_iter = len(self._all_in_one)

        self._genus = None
    
    def _get_bracket(self):
        # max num of eliminations in a bracket 
        s_max = int(np.floor(np.log(self.max_budget / self.min_budget) / np.log(self.eta)))

        # num of downsampling left at stage i in range(s_max + 1)
        n_downsampling = np.linspace(start=s_max, stop=0, num=s_max + 1)

        budgets = (self.max_budget * np.power(self.eta, -n_downsampling)).tolist()
        n_configs = (np.power(self.eta, n_downsampling)).tolist()

        bracket = tuple((int(n), int(b)) for n, b in zip(n_configs, budgets))
        return bracket

    def _init_eval_genus(self, obj : Callable, **kwargs):
        genus = dict()

        for (pop_size, budget) in self._all_in_one:
            species = dict()
            species["population"] = self._init_population(pop_size)
            species["fitness"] = self._eval_population(obj, species["population"], budget, **kwargs)
            
            #genus[budget] = species
            genus[budget] = self._sort_species(species)
        return genus
    
    def _sort_species(self, species : np.ndarray):
        species = species.copy()
        ranking = np.argsort(species["fitness"])
        if self.mode == "max":
            ranking = ranking[::-1]
        
        species["population"] = species["population"][ranking]
        species["fitness"] = species["fitness"][ranking]

        return species

    def _select_promotions(self, target : dict, previous : dict):
        promotions = []
        pop_size = len(target["population"])

        for individual in previous["population"]:
            # If individual already in target, then ignore it to minimize fn_evals
            if np.any(np.all(individual == target["population"], axis=1)):
                continue
            
            promotions.append(individual)
        
        if len(promotions) >= pop_size:
            promotions = promotions[:pop_size]
        else:
            return previous["population"][:pop_size]
            # raise BufferError("Not enough to promote")
            # Can simply pick top pop_size many individuals even 
            # if duplicates exist in target
        
        return np.asarray(promotions)
    
    def _get_alt_population(self, target : dict, previous : dict):

        # stage == 0, previous is None
        if previous is None:
            
            # Edge case where stage==0, but target population too small
            # for vanilla mutation, so we need an alt_pop that is not None
            if len(target) < self._min_pop_size :
                previous = target
            else:
                return None

        pop_size = len(target["population"])
        alt_pop = previous["population"][:pop_size]

        if len(alt_pop) < self._min_pop_size:
            filler_size = self._min_pop_size - len(alt_pop) + 1
            filler_pop = self._sample(self.global_pupulation, filler_size)

            alt_pop = np.concatenate([filler_pop, alt_pop])
        
        return alt_pop
    
    def optimize(self, obj : Callable, limit : int = 10, unit : str = "iter", **kwargs):
        self._start_timer()

        self.genus = self._init_eval_genus(obj, **kwargs)

        try:
            while True: # DEHB iteration
                self._iteration_counter += 1

                for j in range(self._SH_iter): # SH iterations
                    
                    previous = None
                    bracket = self._all_in_one[j:]

                    for stage, (pop_size, budget) in enumerate(bracket): # stages in a bracket
                        if self._is_termination(limit, unit):
                            raise StopIteration
                        
                        target =  self.genus[budget]

                        # Only True for first DEHB iteration and non-inital SH stage
                        promotion = True if self._iteration_counter == 0 and stage > 0 else False
                        if promotion:
                            children = self._select_promotions(target, previous)
                        else:
                            alt_pop = self._get_alt_population(target, previous)
                            children = self._next_generation(target["population"], alt_pop)

                        children_fitness = self._eval_population(obj, children, budget, **kwargs)
                        target["population"], target["fitness"] = self._selection(target["population"], children, target["fitness"], children_fitness)

                        target = self._sort_species(target)
                        self.genus[budget] = target
                        previous = target

        except StopIteration : 
            return self.inc_config
            
    def _init_params(self):
        params = {
            "min_budget" : self.min_budget,
            "max_budget" : self.max_budget,
            "eta"  : self.eta
        }
        params.update(super()._init_params())
        return params

    @property
    def global_pupulation(self):
        assert self.genus is not None, AssertionError("No populaiton in genus initialized")
        return np.concatenate([species["population"] for species in self.genus.values()])


def obj(x : ConfigSpace.Configuration, budget : int, **kwargs):
    """Sample objective function"""
    dataset_id = kwargs["dataset_id"]
    print(x.get_dictionary())

    y = list()
    for name in x:
        if type(x[name]) is str:
            y.append(len(x[name]))
        else:
            y.append(x[name])

    f1_score = np.linalg.norm(y) 
    
    return {"f1_score":f1_score}

if __name__ == "__main__":

    rs = np.random.RandomState(seed=SEED)
    
    space = ConfigVectorSpace(
        name="neuralnetwork",
        seed=SEED,
        # TODO : find distribution for drop_0, drop_1 and reg_const
        # SUGGESTIONS : drop_0 and drop_1 conditional on layer, as deeper layers need larger dropout
        # space={
        #     "lr": ConfigSpace.UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-1, log=True, default_value=1e-3),
        #     "dropout_first": ConfigSpace.Float('dropout_first', bounds=(0, 0.9), default=0.34, distribution=ConfigSpace.Normal(mu=0.5, sigma=0.35)),
        #     "dropout_second": ConfigSpace.Float('dropout_second', bounds=(0, 0.9), default=0.34, distribution=ConfigSpace.Normal(mu=0.5, sigma=0.35)),
        #     #"dropout_1": ConfigSpace.Float('dropout_1', bounds=(0, 0.9), default=0.34, distribution=ConfigSpace.Beta(alpha=2, beta=3)),
        #     "weight_decay": ConfigSpace.Float("weight_decay", bounds=(0, 5), default=0.1),
        #     #"penalty": ConfigSpace.CategoricalHyperparameter("reg_type", choices=["l1", "l2"], weights=[0.5, 0.5], default_value="l2"),
        #     "n_blocks": ConfigSpace.UniformIntegerHyperparameter("n_blocks", lower=2, upper=6, default_value=2),
        #     "batch_size": ConfigSpace.OrdinalHyperparameter("batch_size", sequence=[64, 128, 512], default_value=64),
        #     "d_main": ConfigSpace.OrdinalHyperparameter("d_main", sequence=[32, 64, 128,256,512], default_value=128),
        #     "d_hidden" : ConfigSpace.OrdinalHyperparameter("d_hidden", sequence=[64, 128, 256, 512], default_value=64)
        # },
        space={
            "a": ConfigSpace.UniformFloatHyperparameter("a", lower=-5.0, upper=5.0),
            "b": ConfigSpace.UniformFloatHyperparameter("b", lower=-5.0, upper=5.0),
            "c": ConfigSpace.Categorical("c", ["mouse", "cat", "elephant"], weights=[2, 1, 1])
        },
    )

    dehb = DEHB(space, min_budget=1, max_budget=1000, rs=rs)

    start_time = time.process_time()

    print(f"Best configuration  {dehb.optimize(obj, limit=1,  unit='sec', dataset_id=0)}")
    print(f"Time elapsed (CPU time): {(time.process_time() - start_time):.4f} seconds")
    # dehb.save_data()
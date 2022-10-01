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

    def sample_vectors(self, size: int):
        """Sample from config space 'size' many vectors

        Args:
            size (int): Number of vectors to sample

        Returns:
            list: A list of vectors
        """
        configurations = super().sample_configuration(size)
        if not isinstance(configurations, list):
            configurations = [configurations]

        vectors = [self._to_vector(config) for config in configurations]
        return vectors

    def _to_vector(self, config: ConfigSpace.Configuration) -> np.array:
        """Converts ConfigSpace.Configuration object to numpy array scaled to [0,1]
        Works when self is a ConfigVectorSpace object and the input config is a ConfigSpace.Configuration object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.

        Returns:
            np.ndarray: single configuration vector
        """
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

    def to_config(self, vector: np.ndarray) -> ConfigSpace.Configuration:
        """Converts numpy array to ConfigSpace.Configuration object
        Works when self is a ConfigVectorSpace object and the input vector is in the domain [0, 1].

        Returns:
            ConfigSpace.Configuration: A configuration
        """
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = impute_inactive_values(self.sample_configuration()).get_dictionary()

        # iterates over all hyperparameters and normalizes each based on its type
        for hyper in self.hyperparameters:
            idx = self.name_to_idx[hyper.name]

            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[idx] >= ranges))[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
                param_value = hyper.choices[np.where((vector[idx] >= ranges))[0][-1]]
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
        new_config = deactivate_inactive_hyperparameters(configuration=new_config, configuration_space=self)
        return new_config


class DE(object):
    def __init__(self,
                 space: ConfigVectorSpace,
                 crossover_prob: float = 0.9,
                 mutation_factor: float = 0.8,
                 metric="f1_Score",
                 mode="max",
                 rs: np.random.RandomState = None,
                 bound_control="random",
                 save_path: Union[str, None] = "./data.json",
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
        self._eval_counter = 0
        self._iteration_counter = -1

    def _init_population(self, pop_size: int) -> np.ndarray:
        """Initialize a population with randomly sampled vectors

        Args:
            pop_size (int): Size of a population

        Returns:
            np.ndarray: A 2D vector of shape (pop_size, space.dim)
        """
        # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
        population = self.space.sample_vectors(size=pop_size)
        return np.asarray(population)

    def _eval_population(self, obj: Callable, population: Union[np.ndarray, list], budget: int, **kwargs) -> np.ndarray:
        """Evaluate each candidate in population. The incumbent config is identified.

        Args:
            obj (Callable): The black-box objective function
            population (Union[np.ndarray, list]): A list of configuration vectors
            budget (int): The budget to forward to the objective function obj

        Raises:
            StopIteration: Is handled in self.optimize. Raised when
                the limit of DEHB execution is reached.

        Returns:
            np.ndarray: A vecto of shape (pop_size, 1) with fitess score of
                each evaluated configuration.
        """
        fitness = []

        for candidate in population:
            if self._is_termination():
                raise StopIteration

            config = self.space.to_config(candidate)
            result = obj(config, budget, **kwargs)

            assert isinstance(result, dict), TypeError("Objective function must return a dictionary")
            assert self.metric in result, KeyError(f"Given mteric '{self.metric}' not found in dictionary \
                                                    {result}, returned by the objective function")
            score = result[self.metric]

            condition = {
                "min": score < self.inc_score,
                "max": score > self.inc_score,
            }

            config_dict = config.get_dictionary()
            config_dict.update({"budget": budget})

            if condition[self.mode]:
                self.inc_config = config_dict
                self.inc_score = score

            fitness.append(score)
            # trajectory updated at every fn eval, regardless of save_freq
            self.traj.append(self.inc_score)
            self.runtime.append(self._time_elapsed())
            self._update_history(candidate, result, budget)

            self._eval_counter += 1

            # Update history and wrtie data every 'save_freq' objective fn evaluations
            if self._eval_counter % self.save_freq == 0:
                self.save_data()

        return np.asarray(fitness)

    def _update_history(self, candidate: np.ndarray, result: dict, budget: int):
        """Update the record of evaluated cadidates, results and other info

        Args:
            candidate (np.ndarray): A configuration vector
            result (dict): The dictionray returned by objective function
            budget (int): The budget the vector was executed for
        """
        record = {
            "candidate": candidate.tolist(),
            "budget": budget}

        record.update(result)
        self.histroy.append(record)

    def _sample(self, population: np.ndarray, size: int, replace=False) -> np.ndarray:
        """Sample from an existing population

        Args:
            population (np.ndarray): A 2D array of configuration vectors
            size (int): The number vectors to be sampled
            replace (bool, optional): Whether to sample with replacement or not. Defaults to False.

        Returns:
            nd.ndarray: A subset of pupulation
        """

        selection = self.rs.choice(np.arange(len(population)), size, replace=replace)
        return population[selection]

    def _mutation(self, population: np.ndarray) -> np.ndarray:
        """Perform rand/1 mutation operation

        Args:
            population (np.ndarray): space to sample base vectors of mutant

        Returns:
            np.ndarray: A single mutant vector of shape (space.dim, 1)
        """
        pop_size = len(population)
        assert pop_size > self._min_pop_size, ValueError(f"Population too small ( < DE()._min_pop_size = \
                                                          {self._min_pop_size}) for mutation")

        base, a, b = self._sample(population, self._min_pop_size)

        diff = a - b
        mutant = base + self.mutation_factor * diff
        return mutant

    def _crossover(self, candidate: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform binary crossover by replacing mutant's compenent values with crossover
        probability by candidate value

        Args:
            candidate (np.ndarray): A parent configuration vector
            mutant (np.ndarray): A mutant configuration vector

        Returns:
            np.ndarray: A child of shape (space.dim, 1)
        """
        # Sample a random value from U[0, 1] for every dimension
        p = self.rs.rand(self.space.dim)

        # perform binomial crossover
        child = np.asarray([mutant[i] if p[i] < self.crossover_prob else candidate[i] for i in range(self.space.dim)])
        return child

    def _selection(self, population: np.ndarray, children: np.ndarray,
                   fitness: np.ndarray, children_fitness: np.ndarray):
        """Conduct parent-child competition.

        Args:
            population (np.ndarray): A 2D array of configuration vectors
            children (np.ndarray): A 2D array of configuration vectors corresoinding to
                each parent in population
            fitness (np.ndarray): A fitness score of each parent in population.
            children_fitness (np.ndarray): A fitness score of each child in children.

        Returns:
            (np.ndarray, np.ndarray): New generation of population and corresponding
                fitness scores.
        """
        pop_size = len(population)
        for i in range(pop_size):
            condition = {
                "max": children_fitness[i] >= fitness[i],
                "min": children_fitness[i] <= fitness[i]
                }
            if condition[self.mode]:
                population[i] = children[i]
                fitness[i] = children_fitness[i]

        return population, fitness

    def _check_bounds(self, vector: np.ndarray) -> np.ndarray:
        """Correct for config vector components that exceeds valid range (0,1)

        Args:
            vector (np.ndarray): A configuration vector

        Returns:
            np.ndarray: A probably corrected configuration vector
        """

        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.bound_control == 'random':
            vector[violations] = self.rs.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            # Can be exploited by optimizer if solution at clip limits
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def _next_generation(self, population: np.ndarray, alt_pop: Union[np.ndarray, None] = None):
        """Generate next generation of population through mutation, crossover
            and selection operation.

        Args:
            population (np.ndaray): A 2D array of configuration vectors
            alt_pop (Union[np.ndarray, None], optional): An alternate population
                for sampling base vectors for mutation operation. Defaults to None.

        Returns:
            np.ndarray: A 2D array of evolved population of configuration vectors
        """
        children = []

        for candidate in population:
            # If alt_pop is None, vanilla mutation is perfomed
            mutant = self._mutation(alt_pop if alt_pop is not None else population)
            mutant = self._check_bounds(mutant)
            child = self._crossover(candidate, mutant)
            children.append(child)

        return children

    def optimize(self,
                 obj: Callable,
                 budget=None, pop_size: Union[int, None] = None,
                 limit: int = 10,
                 unit: str = "iter",
                 **kwargs):
        """Perform optimization of objective function subject to the specified limits.

        Args:
            obj (Callable): A black-box objective function
            budget (_type_, optional): A budget to execute the objective function at. Defaults to None.
            pop_size (Union[int,None], optional): size of maintained population. Defaults to None.
            limit (int, optional): optimizaiton stop limit. Defaults to 10.
            unit (str, optional): optimizaiton stop condition. Defaults to "iter".

        Returns:
            _type_: _description_
        """
        self._set_limit(limit, unit)

        if pop_size is None:
            pop_size = 10 * self.space.dim  # heuristic

        try:
            # Initialize and Evaluate the population
            population = self._init_population(pop_size)
            fitness = self._eval_population(obj, population, budget, **kwargs)

            # Until budget is exhausted
            while not self._is_termination(limit, unit):
                self._iteration_counter += 1

                children = self._next_generation(population)

                children_fitness = self._eval_population(obj, children, budget)
                population, fitness = self._selection(population, children, fitness, children_fitness)

        except StopIteration:
            return self.inc_config

    def _set_limit(self, limit: int, unit: str):
        self.limit = limit
        self.unit = unit
        self._start_timer()

    def _start_timer(self):
        self._wall_clock_start = time.time()

    def _time_elapsed(self):
        # time difference in sec
        diff = time.time() - self._wall_clock_start
        return diff

    def _is_termination(self):
        assert self.unit in ["hr", "min", "sec", "evals", "iter"], ValueError("Unrecognized unit given")

        if self.unit in ["hr", "min", "sec"]:
            scale = {
                "sec": 1,
                "min": 60,
                "hr": 60 * 60
            }
            diff = self._time_elapsed()
            return diff >= self.limit * scale[self.unit]
        elif self.unit == "evals":
            return self._eval_counter >= self.limit
        else:
            return self._iteration_counter >= self.limit

    def save_data(self):
        """Save the history of evaluations, incumbent trajectory and other optimization related
            parameters to a json file
        """
        data = {
            "params": self._init_params(),
            "result": {
                "best_config": self.inc_config,
                "best_score": self.inc_score,
            },
            "traj": self.traj,
            "runtime": self.runtime,
            "history": self.histroy,
        }

        if self.save_path is not None:
            # path = os.path.join(self.save_path, "data.json")
            # os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "w+") as outfile:
                json.dump(data, outfile)

    def _init_params(self):
        params = {
            "crossover_prob": self.crossover_prob,
            "mutation_factor": self.mutation_factor,
            "metric": self.metric,
            "mode": self.mode,
            "seed": SEED,
            "bound_control": self.bound_control,
            "iters": self._iteration_counter,
            "evals": self._eval_counter

        }
        return params


class DEHB(DE):
    def __init__(self,
                 space: ConfigVectorSpace,
                 min_budget: int = 10,
                 max_budget: int = 100,
                 eta: int = 2,
                 crossover_prob: float = 0.9,
                 mutation_factor: float = 0.8,
                 metric: str = "f1_score",
                 mode: str = "max",
                 rs: np.random.RandomState = None,
                 bound_control: str = "random",
                 save_path: Union[str, None] = "./data.json",
                 save_freq: int = 10) -> None:

        super().__init__(space=space,
                         crossover_prob=crossover_prob,
                         mutation_factor=mutation_factor,
                         metric=metric,
                         mode=mode,
                         rs=rs,
                         bound_control=bound_control,
                         save_path=save_path,
                         save_freq=save_freq
                         )

        self.min_budget = min_budget
        self.max_budget = max_budget

        self.eta = eta

        self._all_in_one = self._get_bracket()
        self._SH_iter = len(self._all_in_one)

        self._genus = None

    def _get_bracket(self):
        """Compute a full bracket of SH iteration. All brackets
        that will be executed can be extracted as a slice of full bracket.

        Returns:
            tuple: A sequence of tuple of the form ((n_configs, budget_per_config), ...)
        """
        # max num of eliminations in a bracket
        s_max = int(np.floor(np.log(self.max_budget / self.min_budget) / np.log(self.eta)))

        # num of downsampling left at stage i in range(s_max + 1)
        n_downsampling = np.linspace(start=s_max, stop=0, num=s_max + 1)

        budgets = (self.max_budget * np.power(self.eta, -n_downsampling)).tolist()
        n_configs = (np.power(self.eta, n_downsampling)).tolist()

        bracket = tuple((int(n), int(b)) for n, b in zip(n_configs, budgets))
        return bracket

    def _init_eval_genus(self, obj: Callable, **kwargs):
        """Initialize and evaluate the genus. A genus is a dictiontay that holds
        every population and corresponding fitness scores mangaed by DEHB, indexed by budget.

        Args:
            obj (Callable): A black-box objective funciton

        Returns:
            dict: All population with budget as key.
        """
        genus = dict()

        for (pop_size, budget) in self._all_in_one:
            species = dict()
            species["population"] = self._init_population(pop_size)
            species["fitness"] = self._eval_population(obj, species["population"], budget, **kwargs)

            genus[budget] = self._sort_species(species)
        return genus

    def _sort_species(self, species: dict):
        """Sort the species on the basis of fitness score

        Args:
            species (dict): A single population and corresponding fitness

        Returns:
            dict: An ordered populalation and fitness score
        """
        species = species.copy()
        ranking = np.argsort(species["fitness"])
        if self.mode == "max":
            ranking = ranking[::-1]

        species["population"] = species["population"][ranking]
        species["fitness"] = species["fitness"][ranking]

        return species

    def _select_promotions(self, target: dict, previous: dict):
        """Select the top performers from previous species to be promoted as children
        of target species. If individuals in previous already in target, then next best individual
        in previous.

        Args:
            target (dict): A species associated to higher budget
            previous (dict): A species associated to lower budget

        Returns:
            np.ndarray: A 2D array of shape (len(target), shape.dim) with promoted
                configuration vectors.
        """
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

    def _get_alt_population(self, target: dict, previous: dict):
        """To generate an alternate population for sampling base vectors for mutation
        operation.

        Args:
            target (dict): A species associated to higher budget
            previous (dict): A species associated to lower budget

        Returns:
            np.ndarray: A 2D array of shape (len(target), shape.dim), with an
                alternate population
        """
        # stage == 0, previous is None
        if previous is None:

            # Edge case where stage==0, but target population too small
            # for vanilla mutation, so we need an alt_pop that is not None
            if len(target) < self._min_pop_size:
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

    def optimize(self, obj: Callable, limit: int = 10, unit: str = "iter", **kwargs):
        """Optimize the objective function until the specified limit is reached

        Args:
            obj (Callable): _description_
            limit (int, optional): optimizaiton stop limit. Defaults to 10.
            unit (str, optional): optimizaiton stop condition. Defaults to "iter".

        Returns:
            dict: The best configuration yet
        """
        self._set_limit(limit, unit)

        try:
            self.genus = self._init_eval_genus(obj, **kwargs)

            while True:  # DEHB iteration
                self._iteration_counter += 1

                for j in range(self._SH_iter):  # SH iterations

                    previous = None
                    bracket = self._all_in_one[j:]

                    for stage, (pop_size, budget) in enumerate(bracket):  # stages in a bracket
                        target = self.genus[budget]

                        # Only True for first DEHB iteration and non-inital SH stage
                        promotion = True if self._iteration_counter == 0 and stage > 0 else False
                        if promotion:
                            children = self._select_promotions(target, previous)
                        else:
                            alt_pop = self._get_alt_population(target, previous)
                            children = self._next_generation(target["population"], alt_pop)

                        children_fitness = self._eval_population(obj, children, budget, **kwargs)
                        target["population"], target["fitness"] = self._selection(target["population"],
                                                                                  children, target["fitness"],
                                                                                  children_fitness)

                        target = self._sort_species(target)
                        self.genus[budget] = target
                        previous = target

        except StopIteration:
            return self.inc_config

    def _init_params(self):
        params = {
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
            "eta": self.eta
        }
        params.update(super()._init_params())
        return params

    @property
    def global_pupulation(self):
        """Pool population of all species together

        Returns:
            _type_: _description_
        """
        assert self.genus is not None, AssertionError("No populaiton in genus initialized")
        return np.concatenate([species["population"] for species in self.genus.values()])

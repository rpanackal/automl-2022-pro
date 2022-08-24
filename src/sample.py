from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
import constants

cs = ConfigurationSpace(
    name="neuralnetwork",
    seed=constants.SEED,
    space={
        "c": Float("c", bounds=(0.1, 1.5), distribution=Normal(1, 10), log=True),
        "a": Categorical("a", ["mouse", "cat", "dog"], weights=[2, 1, 1]),
        "b": Integer("b", bounds=(2, 10)),
    },
)

configs = cs.sample_configuration(1)

print(configs)
print(cs.get_hyperparameters())

hps = {}
for i, hp in enumerate(cs.get_hyperparameters()):
    # maps hyperparameter name to positional index in vector form
    hps[hp.name] = i
print(hps)
print(configs.get_array())

print(None or 0)

a = {
    "x" : 1,
    "y" : 2,
}

b ={
    "u" : 3,
    "v" : 4
}
a.update(b)
print(a)

import time


limit = 1
wall_clock_start = time.time()
#time.sleep(10)
wall_clock_end = time.time()
diff = wall_clock_end - wall_clock_start

print(diff)
#return diff >= limit

a = {"a":10, "b":20}
l = list(a.keys())

print(a[l])
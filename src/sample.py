from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
import constants

cs = ConfigurationSpace(
    name="neuralnetwork",
    seed=constants.SEED,
    space={
        "a": Float("a", bounds=(0.1, 1.5), distribution=Normal(1, 10), log=True),
        "b": Integer("b", bounds=(2, 10)),
        "c": Categorical("c", ["mouse", "cat", "dog"], weights=[2, 1, 1]),
    },
)

configs = cs.sample_configuration(2)

print(configs)
print(cs.get_hyperparameters())

hps = {}
for i, hp in enumerate(cs.get_hyperparameters()):
    # maps hyperparameter name to positional index in vector form
    hps[hp.name] = i

print(hps)
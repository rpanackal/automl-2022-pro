import openml
import numpy as np
import math
from sklearn import preprocessing
from config import SEED


def dataset_num_to_id(dataset_number):
    """Convert number from 0 to 10 for easier looping over datasets

        Parameters
        ----------
        dataset_number: int
            Number to predefined id of the dataset (id from openml)

        Returns
        -------
        id of the dataset
        """
    dataset_ids = [40590, 40591, 40592, 40593, 40594, 40595, 40596, 40597, 40588, 40589]
    return dataset_ids[dataset_number]


class OpenMlDataset():
    def __init__(self, dataset_id, threshold=0.9, normalization_type="standard", split=0.75) -> None:
        """Class will load dataset and preprocess predictors/attributes

        Parameters
        ----------
        dataset_id: int
            Id of specific dataset from Openml

        threshold: float
            Threshold for correlation. If this threshold is exceeded by abs(correlation_score),
            then remove attirbute.

        normalization_type: "standard" or "robust"
            How to normalize numeric attributes. Standard will set average to 0 and std to 1.
            Robust will use remove median and scale according to quantile range.

        split: float
            Value between 0.0-1.0. It is used to determine what % of the whole
            data will be used as training. Rest will be used as test data.

        Returns
        -------
        """
        self.dataset_id = dataset_id
        self.random_generator = np.random.RandomState(SEED)
        self.split = split
        self.load_dataset()
        self.pred_preprocessing = PredictorsPreprocessingModule(self.pred_data, self.pred_categorical_indicator,
                                                                correlation_threshold=threshold,
                                                                normalization_type=normalization_type)
        self.pred_data = self.pred_preprocessing.get_data()
        self.train_test_split()

    def load_dataset(self):
        """Load the dataset from website and split dataset into train and test subsets

        Parameters
        ----------

        Returns
        -------
        """
        dataset = openml.datasets.get_dataset(self.dataset_id)
        self.targets_name = dataset.default_target_attribute.split(",")

        X, _, categorical_indicator, features_name = dataset.get_data(dataset_format="dataframe")
        self.features_name = features_name
        self.categorical_indicator = categorical_indicator

        for i, name in enumerate(self.features_name):
            if not self.categorical_indicator[i]:
                continue
            encoder = preprocessing.LabelEncoder()
            encoder.fit(X[name])
            column = encoder.transform(X[name])
            X.loc[:, name] = column.astype('float32')

        X = X.astype('float32')
        self.data = X.to_numpy()

        print("Chosen dataset is ", dataset.name, " (id:"+str(self.dataset_id)+")")
        print("Original dataset shape is ", self.data.shape)
        print("Number of predictors : ", len(self.features_name) - len(self.targets_name),
              "Number of targets : ", len(self.targets_name))

        # Split data into predictors and targets
        target_inds = []
        pred_inds = []
        for i in range(self.data.shape[1]):
            if self.features_name[i] in self.targets_name:
                target_inds.append(i)
            else:
                pred_inds.append(i)
        self.target_categorical_indicator = [self.categorical_indicator[i] for i in target_inds]
        self.pred_categorical_indicator = [self.categorical_indicator[i] for i in pred_inds]
        self.pred_data = self.data[:, pred_inds]
        self.target_data = self.data[:, target_inds]

    def get_predictors(self):
        """Return training predictors and categorical indicator

        Parameters
        ----------

        Returns
        -------
        train_pred_data: numpy.array(samples, attributes)
            Training predictors

        pred_categorical_indicator: list[Bool]
            List of True/False indicator, which parameter is numeric or categorical.
            Length is the same as training_pred_data.shape[1].
        """
        return self.train_pred_data, self.pred_categorical_indicator

    def get_targets(self):
        """Return training targets and categorical indicator

        Parameters
        ----------

        Returns
        -------
        train_pred_data: numpy.array(samples, attributes)
            Training targets

        pred_categorical_indicator: list[Bool]
            List of True/False indicator, which parameter is numeric or categorical.
            Length is the same as training_tar_data.shape[1].
        """
        return self.train_tar_data, self.target_categorical_indicator

    def get_test_predictors(self):
        """Return test predictors and categorical indicator

        Parameters
        ----------

        Returns
        -------
        test_pred_data: numpy.array(samples, attributes)
            Test predictors

        pred_categorical_indicator: list[Bool]
            List of True/False indicator, which parameter is numeric or categorical.
            Length is the same as test_pred_data.shape[1].
        """
        return self.test_pred_data, self.pred_categorical_indicator

    def get_test_targets(self):
        """Return test targets and categorical indicator

        Parameters
        ----------

        Returns
        -------
        train_pred_data: numpy.array(samples, attributes)
            Test targets

        pred_categorical_indicator: list[Bool]
            List of True/False indicator, which parameter is numeric or categorical.
            Length is the same as test_tar_data.shape[1].
        """
        return self.test_tar_data, self.target_categorical_indicator

    def get_dataset(self):
        """Return whole dataset

        Parameters
        ----------

        Returns
        -------
        pred_data: numpy.array(samples, attributes)
            All predictors

        target_data: numpy.array(samples, attributes)
            All targets
        """
        return self.pred_data, self.target_data

    def train_test_split(self):
        """Split dataset into train and test subset based of the split parameter

        Parameters
        ----------

        Returns
        -------
        """
        perm_idxs = np.arange(self.data.shape[0])
        self.random_generator.shuffle(perm_idxs)
        split_idx = int(self.data.shape[0]*self.split)
        train_idxs = perm_idxs[:split_idx]
        valid_idxs = perm_idxs[split_idx:]
        self.train_pred_data = self.pred_data[train_idxs]
        self.train_tar_data = self.target_data[train_idxs]
        self.test_pred_data = self.pred_data[valid_idxs]
        self.test_tar_data = self.target_data[valid_idxs]


class PredictorsPreprocessingModule():
    def __init__(self, data, categorical_indicator, correlation_threshold=0.8,
                 normalization_type="standard") -> None:
        """Class will preprocess predictors/attributes

        Parameters
        ----------
        data: numpy.array(sample,attributes)
            Predictors/Attributes of the dataset to preprocess.

        correlation_threshold: float
            Threshold for correlation. If this threshold is exceeded by abs(correlation_score), then remove attirbute.

        normalization_type: "standard" or "robust"
            How to normalize numeric attributes. Standard will set average to 0 and std to 1.
            Robust will use remove median and scale according to quantile range.

        Returns
        -------
        """
        self.data = data
        self.categorical_indicator = categorical_indicator
        self.correlation_threshold = correlation_threshold
        self.normalization_type = normalization_type
        assert self.data.shape[1] == len(self.categorical_indicator),   """ Categorical indicator should have
                                                                            length equal to the number of features"""
        self.features_to_remove_list = []
        self.scaler_list = []
        self.preprocessing()

    def preprocessing(self):
        """Function will call all the tests in the correct order

        Parameters
        ----------

        Returns
        -------
        """
        self.transform_2_value_numeric_to_binary()
        self.check_for_single_value_parameters()
        self.check_for_values_with_one_occurance()
        self.check_for_pseudocorrelation()
        self.check_for_correlation()
        # Must be the last
        self.normalize_dataset()
        self.dataset_one_hot_encoding()

    def transform_2_value_numeric_to_binary(self):
        """ Function will over all numeric attributes and convert them to binary
            if particular parameter has only 2 unique values.

        Parameters
        ----------

        Returns
        -------
        """
        transformed_features_num = 0
        for i in range(self.data.shape[1]):
            if self.categorical_indicator[i]:
                pass
            unique_values = np.unique(self.data[:, i])
            if unique_values.shape[0] == 2:
                encoder = preprocessing.LabelEncoder()
                encoder.fit(self.data[:, i])

                self.data[:, i] = (encoder.transform(self.data[:, i])).astype('float32')
                self.categorical_indicator[i] = True
                transformed_features_num = transformed_features_num + 1

        print(transformed_features_num, " numeric features were transformed into binary")

    def check_for_values_with_one_occurance(self):
        """ Function will look for categorical attributes, which has one occurence
        of particular value. It will be remove this attribute.

        Parameters
        ----------

        Returns
        -------
        """
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            if not self.categorical_indicator[i]:
                continue
            temp = (self.data[:, i]).astype(int)
            temp = temp + abs(int(np.amin(temp)))+1
            occurances = np.bincount(temp)
            if 1 in occurances:
                params_to_delete.append(i)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ", old - new, " parameters with 1 occurance of value")

    def check_for_single_value_parameters(self):
        """ Function will look for attribute with one unique value. This attribute is seen as
        just a constant, so it can be removed.

        Parameters
        ----------

        Returns
        -------
        """
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            unique_values = np.unique(self.data[:, i])
            if unique_values.shape[0] == 1:
                params_to_delete.append(i)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ", old - new, " parameters with one unique value")

    def check_for_pseudocorrelation(self):
        """ Function will look for pairs of 2 categorical attributes, which are pseudo-correlated.
            If the number of unique pairs is the same as number of unique values of one attribute
            (with higher number out of two), then remove other attribute. It means, that one value
            in pair can be seen as constant and doesn't provide more information.

        Parameters
        ----------

        Returns
        -------
        """
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            if not self.categorical_indicator[i]:
                continue
            for j in np.arange(i-1, -1, -1):
                if not self.categorical_indicator[j]:
                    continue
                # Check for pseudocorrelation
                temp_i = self.data[:, i]
                temp_j = self.data[:, j]
                uniq_val_num_i = np.unique(self.data[:, i]).shape[0]
                uniq_val_num_j = np.unique(self.data[:, j]).shape[0]
                if uniq_val_num_i > uniq_val_num_j:
                    lower = j
                else:
                    lower = i
                temp_i = temp_i * uniq_val_num_j
                sum = temp_i + temp_j
                uniq_val_sum = np.unique(sum)
                if uniq_val_sum.shape[0] == uniq_val_num_i or uniq_val_sum.shape[0] == uniq_val_num_j:
                    if not (lower in params_to_delete):
                        params_to_delete.append(lower)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ", old - new, " parameters with pseudocorrelated values")

    def check_for_correlation(self):
        """ Function will look for pair of 2 numeric features. If the absolute value
            of their correlation score is than threshold, then removed parameter with less unique values.

        Parameters
        ----------

        Returns
        -------
        """
        params_to_delete = []
        for i in range(self.data.shape[1]):
            if self.categorical_indicator[i]:
                continue
            for j in range(i+1, self.data.shape[1]):
                if self.categorical_indicator[j]:
                    continue

                sum_i = np.sum(self.data[:, i])
                sum_j = np.sum(self.data[:, j])
                sum_i2 = np.sum(self.data[:, i]**2)
                sum_j2 = np.sum(self.data[:, j]**2)
                sum_ij = np.sum(self.data[:, i] * self.data[:, j])
                n = self.data.shape[0]

                correlation = (n*sum_ij - sum_i * sum_j)/(math.sqrt((n * sum_i2 -
                                                                     (sum_i)**2)*(n * sum_j2 - (sum_j)**2)))
                correlation = abs(correlation)
                if correlation >= self.correlation_threshold:
                    if np.unique(self.data[:, i]).shape[0] >= np.unique(self.data[:, j]).shape[0]:
                        if not (j in params_to_delete):
                            params_to_delete.append(j)
                    else:
                        if not (i in params_to_delete):
                            params_to_delete.append(i)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ", old - new, " parameters with correlated values")

    def clear_feature_data(self, indices: list):
        """Function will remove attributes based on the provided list. Shouldn't be used outside class.

        Parameters
        ----------
        indices: list[int]
            Function will remove all attributes pointed by this list. List shouldn't include any duplicates.

        Returns
        -------
        """
        indices.sort()
        self.features_to_remove_list.append(indices)
        indices = indices[::-1]
        for i in indices:
            self.categorical_indicator.pop(i)

    def normalize_dataset(self):
        """ Function will normalize all numeric attributes with standard or robust method.
            Decision is based on self.normalization_type.

        Parameters
        ----------

        Returns
        -------
        """
        for i in range(self.data.shape[1]):
            if self.categorical_indicator:
                self.scaler_list.append(None)
                continue

            if self.normalization_type == "standard":
                self.scaler_list.append(preprocessing.StandardScaler())
                self.scaler_list[i].fit(self.data[:, i].reshape(-1, 1))
                self.data[:, i] = self.scaler_list[i].transform(self.data[:, i].reshape(-1, 1)).reshape((-1))
            elif self.normalization_type == "robust":
                self.scaler_list.append(preprocessing.RobustScaler())
                self.scaler_list[i].fit(self.data[:, i].reshape(-1, 1))
                self.data[:, i] = self.scaler_list[i].transform(self.data[:, i].reshape(-1, 1)).reshape((-1))
            else:
                modes = ["standard", "robust"]
                print("Wrong normalization mode. Choose one of the following: ", modes)

    def one_hot_encoding(self, x):
        """ Function will convert categorical attribute into one-hot encoding.
            Binary attributes will be coded using 1 neuron. The rest will be encoded using
            n neurons, where n is the number of unique values.

        Parameters
        ----------

        Returns
        -------
        """
        length = x.shape[0]
        max_value = int(np.max(x))
        new_array = np.zeros((length, max_value+1))
        new_array[np.arange(length), x.astype(int)] = 1.0
        return new_array

    def dataset_one_hot_encoding(self):
        """ Function will convert categorical attributes into one-hot encoding.
            Binary attributes will be coded using 1 neuron. The rest will be encoded using
            n neurons, where n is the number of unique values.

        Parameters
        ----------

        Returns
        -------
        """
        for i in range(self.data.shape[1]-1, -1, -1):
            if np.unique(self.data[:, i]).shape[0] == 2 or not self.categorical_indicator[i]:
                temp_ar = np.reshape(self.data[:, i], (-1, 1))
            else:
                temp_ar = self.one_hot_encoding(self.data[:, i])

            if i+1 == self.data.shape[1]:
                self.data = np.concatenate((self.data[:, :i], temp_ar), axis=1)
            else:
                self.data = np.concatenate((self.data[:, :i], temp_ar, self.data[:, i+1:]), axis=1)

    def get_data(self):
        """Function will return preprocessed dataset
        Parameters
        ----------
        Returns
        -------
        data: numpy.array(samples, attributes)
            Preprocessed dataset
        """
        return self.data


class CrossValidation():
    def __init__(self, pred_data, target_data, split_num) -> None:
        """Create easy to use object for cross validation. This class can be used with Enumerate to get i, (x,y).

        Parameters
        ----------
        pred_data: np.array(samples, attributes)
            predictors/attributes of the dataset
        target_data: np.array(samples, labels)
            targets/labels of the dataset
        split_num: int
            Dataset will be divided in split_num splits

        Returns
        -------
        """
        self.pred_data = pred_data
        self.target_data = target_data
        self.split_num = split_num
        self.make_splits()
        self.iter = 0

    def make_splits(self):
        """Split predictors and targets into splits

        Parameters
        ----------

        Returns
        -------
        """
        samples_per_split = self.pred_data.shape[0]//self.split_num + 1
        self.pred_data_per_split = []
        self.target_data_per_split = []
        for i in range(self.split_num):
            self.pred_data_per_split.append(self.pred_data[i*samples_per_split:min((i+1)*samples_per_split,
                                                                                   self.pred_data.shape[0])])
            self.target_data_per_split.append(self.target_data[i*samples_per_split:min((i+1)*samples_per_split,
                                                                                       self.target_data.shape[0])])

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == self.split_num:
            self.iter = 0
            raise StopIteration

        valid_pred = self.pred_data_per_split[self.iter]
        valid_target = self.target_data_per_split[self.iter]
        train_pred = None
        train_target = None
        for i in range(self.split_num):
            if i == self.iter:
                continue
            if train_pred is None:
                train_pred = self.pred_data_per_split[i]
                train_target = self.target_data_per_split[i]
            else:
                train_pred = np.concatenate((train_pred, self.pred_data_per_split[i]), axis=0)
                train_target = np.concatenate((train_target, self.target_data_per_split[i]), axis=0)
        self.iter = self.iter + 1
        return train_pred, train_target, valid_pred, valid_target

    def __getitem__(self, index):
        valid_pred = self.pred_data_per_split[index]
        valid_target = self.target_data_per_split[index]
        train_pred = None
        train_target = None
        for i in range(self.split_num):
            if i == index:
                continue
            if train_pred is None:
                train_pred = self.pred_data_per_split[i]
                train_target = self.target_data_per_split[i]
            else:
                train_pred = np.concatenate((train_pred, self.pred_data_per_split[i]), axis=0)
                train_target = np.concatenate((train_target, self.target_data_per_split[i]), axis=0)
        return train_pred, train_target, valid_pred, valid_target

    def __len__(self):
        return self.split_num


class Split:
    def __init__(self, x, y):
        self.X = x
        self.y = y


dataset_names = {40588: "birds",
                 40589: "emotions",
                 40590: "enron",
                 40591: "genbase",
                 40592: "image",
                 40593: "langLog",
                 40594: "reuters",
                 40595: "scene",
                 40596: "slashdot",
                 40597: "yeast"}

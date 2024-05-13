from typing import List, Tuple, Dict, Callable

import os
import shutil
import shelve

import numpy as np
from numpy import savetxt
from numpy.typing import NDArray
from math import floor, sqrt, ceil
from matplotlib import pyplot

from keras import Sequential, Model
from keras.callbacks import History
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam, SGD

import objgraph

class Configuration:

    def __init__(self, sample_size: int, layer_descriptor: List[int],  activation_descriptor: List[str], learning_rate: float, dropout_rate: float):
        self.sample_size = sample_size
        self.layer_descriptor = layer_descriptor
        self.activation_descriptor = activation_descriptor
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def as_tuple(self) -> Tuple[int, Tuple[int, ...], Tuple[str, ...], float, float]:
        return self.sample_size, tuple(self.layer_descriptor), tuple(self.activation_descriptor), self.learning_rate, self.dropout_rate

class GridSearch:

    KEY_STATE = "state"

    KEY_INPUT_DIM = "input_dim"

    KEY_LOSS_FUNCTION = "loss_function"

    KEY_ITERATIONS = "iterations"

    KEY_TEST_SPLIT = "test_split"

    KEY_BATCH_SIZE = "batch_size"

    KEY_EPOCHS = "epochs"

    KEY_MIN_SAMPLE_SIZE_EPX = "min_sample_size_exp"

    KEY_MAX_SAMPLE_SIZE_EPX = "max_sample_size_exp"

    KEY_LAYER_DESCRIPTOR_LIST = "layer_descriptor_list"

    KEY_ACTIVATION_DESCRIPTOR_LIST = "activation_descriptor_list"

    KEY_LEARNING_RATE_LIST = "learning_rate_list"

    KEY_DROPOUT_RATE_LIST = "dropout_rate_list"

    KEY_EXPLORED_SET = "explored_set"

    STATE_VALUE_CREATED = "created"

    STATE_VALUE_FIXED_PARAMS_INITIALIZED = "fixed_params_initialized"

    STATE_VALUE_SEARCH_SPACE_INITIALIZED = "search_space_initialized"

    STATE_VALUE_SEARCHING = "searching"

    STATE_VALUE_PAUSED = "paused"

    STATE_VALUE_COMPLETED = "completed"

    METRICS = ["accuracy"]

    # Reviewed

    @staticmethod
    def create_dir_if_not_existent(path: str) -> str:
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    # Reviewed
    @staticmethod
    def recreate_dir_if_existent(path: str) -> str:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return path

    # Reviewed
    @staticmethod
    def extract_training_metrics(history: History) -> Dict[str, List[float]]:
        return history.history

    # TODO
    @staticmethod
    def extract_evaluation_metrics(eval_results, model: Model) -> Dict[str, float]:
        contained_eval_results = []
        for result in eval_results:
            contained_eval_results.append([result])
        return dict(zip(model.metrics_names, contained_eval_results))

    # Reviewed
    @staticmethod
    def create_architecture(input_dim: int, units_list: List[int], activation_list: List[str],dropout_rate: float) -> Model:
        model = Sequential()
        model.add(Dense(units_list[0], activation=activation_list[0], input_dim=input_dim))
        for layer_index in range(1, len(units_list)):
            model.add(Dense(units_list[layer_index], activation=activation_list[layer_index]))
            if dropout_rate is not None:
                model.add(Dropout(dropout_rate))
        model.summary()
        return model

    # TODO
    @staticmethod
    def average_training_metrics(aggregated_training_metrics: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
        averaged_training_data_metrics = dict()
        for metric in aggregated_training_metrics:
            metric_value_arr = np.array(aggregated_training_metrics[metric])
            averaged_training_data_metrics[metric] = metric_value_arr.mean(axis=0).tolist()
        return averaged_training_data_metrics

    # TODO
    @staticmethod
    def average_test_metrics(aggregated_training_metrics: Dict[str, List[float]]) -> Dict[str, float]:
        averaged_test_data_metrics = dict()
        for metric in aggregated_training_metrics:
            metric_value_arr = np.array(aggregated_training_metrics[metric])
            if len(metric_value_arr) > 0:
                averaged_test_data_metrics[metric] = metric_value_arr.mean(axis=0).tolist()[0]
            else:
                averaged_test_data_metrics[metric] = None
        return averaged_test_data_metrics

    # Reviewed
    @staticmethod
    def split(x_data: NDArray, y_data: NDArray, split_factor: float) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        max_index = floor(x_data.shape[0] * (1 - split_factor))
        return x_data[:max_index], y_data[:max_index], x_data[max_index:], y_data[max_index:]

    # Reviewed
    @staticmethod
    def evaluate_data(y_data: NDArray[int], labeling_func: Callable[[NDArray[int]], str], frequency_dict: Dict[str, int]) -> None:
        for row in y_data:
            label = labeling_func(row)
            if label not in frequency_dict:
                frequency_dict[label] = 0
            frequency_dict[label] = frequency_dict[label] + 1

    # Reviewed
    @staticmethod
    def save_as_barchart(data_dict, path: str, name: str, xLabel: str, yLabel: str, title: str) -> None:
        sorted_labels = sorted(list(data_dict.keys()))
        sorted_values = [data_dict[key] for key in sorted_labels]
        pyplot.bar(sorted_labels, sorted_values)

        #pyplot.rcParams["figure.figsize"] = [7.50, 3.50]
        #pyplot.rcParams["figure.autolayout"] = True

        pyplot.xlabel(xLabel)
        pyplot.ylabel(yLabel)
        pyplot.title(title)
        pyplot.xticks(rotation=45)  # Rotates X-Axis Ticks by 45-degrees

        #spacing = 0.100
        #pyplot.figure().subplots_adjust(bottom=spacing)

        pyplot.savefig(GridSearch.create_dir_if_not_existent(path) + "/" + name)
        pyplot.close()

    def __init__(self, name):
        if name is None or name == "":
            raise ValueError()
        self.do_pause = False
        self.base_dir = GridSearch.create_dir_if_not_existent("./data/" + name)
        self.search_db = shelve.open(self.base_dir + "/" + name + "_persistence", writeback=True)
        if GridSearch.KEY_EXPLORED_SET not in self.search_db:
            self.search_db[GridSearch.KEY_EXPLORED_SET] = []
        if GridSearch.KEY_STATE not in self.search_db:
            self.search_db[GridSearch.KEY_STATE] = GridSearch.STATE_VALUE_CREATED
        self.search_db.sync()

    def get_state(self) -> str:
        return self.search_db[GridSearch.KEY_STATE]

    def get_input_dim(self) -> int:
        return self.search_db[GridSearch.KEY_INPUT_DIM]

    def get_loss_function(self) -> str:
        return self.search_db[GridSearch.KEY_LOSS_FUNCTION]

    def get_iterations(self) -> int:
        return self.search_db[GridSearch.KEY_ITERATIONS]

    def get_test_split(self) -> float:
        return self.search_db[GridSearch.KEY_TEST_SPLIT]

    def get_batch_size(self) -> int:
        return self.search_db[GridSearch.KEY_BATCH_SIZE]

    def get_epochs(self) -> int:
        return self.search_db[GridSearch.KEY_EPOCHS]

    def get_configuration_dir(self, configuration: Configuration) -> str:
        return self.base_dir + "/{0}_{1}_{2}_{3}_{4}".format(configuration.sample_size, configuration.layer_descriptor, configuration.activation_descriptor, configuration.learning_rate, configuration.dropout_rate)

    def get_iteration_dir(self, configuration: Configuration, iteration: int) -> str:
        return self.get_configuration_dir(configuration) + "/" + str(iteration)

    def set_fixed_parameters(self, input_dim: int, loss_function: str, iterations: int = 1, test_split: float = 0.2,  batch_size: int=64, epochs: int=20) -> None:
        if self.get_state() == GridSearch.STATE_VALUE_CREATED:
            self.search_db[GridSearch.KEY_INPUT_DIM] = input_dim
            self.search_db[GridSearch.KEY_LOSS_FUNCTION] = loss_function
            self.search_db[GridSearch.KEY_ITERATIONS] = iterations
            self.search_db[GridSearch.KEY_TEST_SPLIT] = test_split
            self.search_db[GridSearch.KEY_BATCH_SIZE] = batch_size
            self.search_db[GridSearch.KEY_EPOCHS] = epochs
            self.search_db[GridSearch.KEY_STATE] = GridSearch.STATE_VALUE_FIXED_PARAMS_INITIALIZED
            self.search_db.sync()

    def set_search_space(self, min_sample_size_2_exp:int , max_sample_size_2_exp: int, layer_descriptor_list: List[List[int]], activation_descriptor_list: List[List[str]], learning_rate_list: List[float], dropout_rate_list: List[float]) -> None:
        if self.get_state() == GridSearch.STATE_VALUE_FIXED_PARAMS_INITIALIZED:
            self.search_db[GridSearch.KEY_MIN_SAMPLE_SIZE_EPX] = min_sample_size_2_exp
            self.search_db[GridSearch.KEY_MAX_SAMPLE_SIZE_EPX] = max_sample_size_2_exp
            self.search_db[GridSearch.KEY_LAYER_DESCRIPTOR_LIST] = layer_descriptor_list
            self.search_db[GridSearch.KEY_ACTIVATION_DESCRIPTOR_LIST] = activation_descriptor_list
            self.search_db[GridSearch.KEY_LEARNING_RATE_LIST] = learning_rate_list
            self.search_db[GridSearch.KEY_DROPOUT_RATE_LIST] = dropout_rate_list
            self.search_db[GridSearch.KEY_STATE] = GridSearch.STATE_VALUE_SEARCH_SPACE_INITIALIZED
            self.search_db.sync()
    def search(self, sample: Callable[[int], Tuple[NDArray, NDArray]], labeling_func: Callable[[NDArray[int]], str]) -> None:
        if self.get_state() != GridSearch.STATE_VALUE_SEARCH_SPACE_INITIALIZED and self.get_state() != GridSearch.STATE_VALUE_PAUSED and self.get_state() != GridSearch.STATE_VALUE_SEARCHING:
            raise ValueError()
        self.do_pause = False
        self.search_db[GridSearch.KEY_STATE] = GridSearch.STATE_VALUE_SEARCHING
        configuration_list: List[Configuration] = list(self.next_configuration())
        conf_index = 0
        while not self.do_pause and conf_index < len(configuration_list):
            cur_configuration = configuration_list[conf_index]
            if not self.is_explored(cur_configuration):
                self.save_configuration(cur_configuration)
                aggregated_training_metrics: Dict[str, List[List[float]]] = dict()
                aggregated_test_metrics: Dict[str, List[float]] = dict()
                configuration_y_training_data_freq: Dict[str, float] = dict()
                configuration_y_validation_data_freq: Dict[str, float] = dict()
                configuration_y_test_data_freq: Dict[str, float] = dict()
                y_training_data_size = 0
                y_validation_data_size = 0
                y_test_data_size = 0
                for iteration in range(0, self.get_iterations()):
                    x_training_data, y_training_data, x_validation_data, y_validation_data, x_test_data, y_test_data = self.create_and_save_training_data(sample, cur_configuration, iteration)
                    y_training_data_size += len(y_training_data)
                    y_validation_data_size += len(y_validation_data)
                    y_test_data_size += len(y_test_data)
                    iteration_y_training_data_freq: Dict[str, int] = dict()
                    iteration_y_validation_data_freq: Dict[str, int] = dict()
                    iteration_y_test_data_freq: Dict[str, int] = dict()
                    GridSearch.evaluate_data(y_training_data, labeling_func, configuration_y_training_data_freq)
                    GridSearch.evaluate_data(y_validation_data, labeling_func, configuration_y_validation_data_freq)
                    GridSearch.evaluate_data(y_test_data, labeling_func, configuration_y_test_data_freq)
                    # The following lines of code evaluate the data driving the current iteration. The results are
                    # plotted and stored in the current iteration's directory.
                    GridSearch.evaluate_data(y_training_data, labeling_func, iteration_y_training_data_freq)
                    GridSearch.evaluate_data(y_validation_data, labeling_func, iteration_y_validation_data_freq)
                    GridSearch.evaluate_data(y_test_data, labeling_func, iteration_y_test_data_freq)
                    GridSearch.save_as_barchart(iteration_y_training_data_freq, self.get_iteration_dir(cur_configuration, iteration) + "/data", "y_training_data_label_freq.png", "", "Frequency", "Sample Size n = {0}".format(len(y_training_data)))
                    GridSearch.save_as_barchart(iteration_y_validation_data_freq, self.get_iteration_dir(cur_configuration, iteration) + "/data", "y_validation_data_label_freq.png", "", "Frequency", "Sample Size n = {0}".format(len(y_validation_data)))
                    GridSearch.save_as_barchart(iteration_y_test_data_freq, self.get_iteration_dir(cur_configuration, iteration) + "/data", "y_test_data_label_freq.png", "", "Frequency", "Sample Size n = {0}".format(len(y_test_data)))
                    model, history = self.train_and_save_model(cur_configuration, iteration, x_training_data, y_training_data, x_validation_data, y_validation_data)
                    eval_results = model.evaluate(x_test_data, y_test_data, verbose=0)
                    training_metrics: Dict[str, List[float]] = GridSearch.extract_training_metrics(history)
                    evaluation_metrics: Dict[str, float] = GridSearch.extract_evaluation_metrics(eval_results, model)
                    self.save_training_metrics(training_metrics, cur_configuration, iteration)
                    self.save_test_metrics(evaluation_metrics, cur_configuration, iteration)
                    for key in training_metrics:
                        if key not in aggregated_training_metrics:
                            aggregated_training_metrics[key] = []
                        aggregated_training_metrics[key].append(training_metrics[key])
                    for key in evaluation_metrics:
                        if key not in aggregated_test_metrics:
                            aggregated_test_metrics[key] = []
                        aggregated_test_metrics[key].append(evaluation_metrics[key])
                for key in configuration_y_training_data_freq:
                    configuration_y_training_data_freq[key] = configuration_y_training_data_freq[key] / self.get_iterations()
                for key in configuration_y_validation_data_freq:
                    configuration_y_validation_data_freq[key] = configuration_y_validation_data_freq[key] / self.get_iterations()
                for key in configuration_y_test_data_freq:
                    configuration_y_test_data_freq[key] = configuration_y_test_data_freq[key] / self.get_iterations()
                GridSearch.save_as_barchart(configuration_y_training_data_freq, self.get_configuration_dir(cur_configuration) + "/results", "avg_y_training_data_label_freq.png", "", "Frequency", "Average Sample Size n = {0}".format(y_training_data_size / self.get_iterations()))
                GridSearch.save_as_barchart(configuration_y_validation_data_freq, self.get_configuration_dir(cur_configuration) + "/results", "avg_y_validation_data_label_freq.png", "", "Frequency", "Average Sample Size n = {0}".format(y_validation_data_size / self.get_iterations()))
                GridSearch.save_as_barchart(configuration_y_test_data_freq, self.get_configuration_dir(cur_configuration) + "/results", "avg_y_test_data_label_freq.png", "", "Frequency","Average Sample Size n = {0}".format(y_test_data_size / self.get_iterations()))
                self.save_averaged_training_metrics(GridSearch.average_training_metrics(aggregated_training_metrics), cur_configuration)
                self.save_averaged_test_metrics(GridSearch.average_test_metrics(aggregated_test_metrics), cur_configuration)
                self.add_to_explored_set(cur_configuration)
            conf_index += 1

    # Tested
    def save_configuration(self, configuration: Configuration) -> None:
        configuration_dir: str = GridSearch.recreate_dir_if_existent(self.get_configuration_dir(configuration))
        f = open(configuration_dir + "/configuration.txt", "a")
        f.write("sample_size={0}\nlayer_description={1}\nactivation_description={2}\nlearning_rate={3}\ndropout_rate={4}\n".format(configuration.sample_size, configuration.layer_descriptor, configuration.activation_descriptor, configuration.learning_rate, configuration.dropout_rate))
        f.close()

    #Reviewed
    def create_and_save_training_data(self, sample: Callable[[int], Tuple[NDArray, NDArray]], configuration: Configuration, iteration: int) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        iteration_dir: str = GridSearch.recreate_dir_if_existent(self.get_iteration_dir(configuration, iteration))
        x_data, y_data, label_name_dict = sample(ceil(configuration.sample_size / (1-self.get_test_split())**2))
        x_training_data, y_training_data, x_test_data, y_test_data = GridSearch.split(x_data, y_data, self.get_test_split())
        x_training_data, y_training_data, x_validation_data, y_validation_data = GridSearch.split(x_training_data, y_training_data, self.get_test_split())
        GridSearch.create_dir_if_not_existent(iteration_dir + "/data")
        savetxt(iteration_dir + "/data/x_training_data", x_training_data, delimiter=",")
        savetxt(iteration_dir + "/data/y_training_data", y_training_data, delimiter=",")
        savetxt(iteration_dir + "/data/x_validation_data", x_validation_data, delimiter=",")
        savetxt(iteration_dir + "/data/y_validation_data", y_validation_data, delimiter=",")
        savetxt(iteration_dir + "/data/x_test_data", x_test_data, delimiter=",")
        savetxt(iteration_dir + "/data/y_test_data", y_test_data, delimiter=",")
        f = open(iteration_dir + "/data/labels.txt", "a")
        for label in label_name_dict:
            f.write(str(label) + ": " + str(label_name_dict[label]) + "\n")
        f.close()
        return x_training_data, y_training_data, x_validation_data, y_validation_data, x_test_data, y_test_data

    def train_and_save_model(self, configuration: Configuration, iteration: int,  x_training_data: NDArray, y_training_data: NDArray, x_validation_data: NDArray, y_validation_data: NDArray) -> (Model, History):
        model: Model = GridSearch.create_architecture(self.get_input_dim(), configuration.layer_descriptor, configuration.activation_descriptor, configuration.dropout_rate)
        model.compile(optimizer=SGD(learning_rate=configuration.learning_rate), loss=self.get_loss_function(), metrics=GridSearch.METRICS)
        history = model.fit(x=x_training_data, y=y_training_data, validation_data=(x_validation_data, y_validation_data), batch_size=self.get_batch_size(), epochs=self.get_epochs(), verbose=0)
        model_path: str = GridSearch.create_dir_if_not_existent(self.get_iteration_dir(configuration, iteration)) + "/model.keras"
        model.save(filepath=model_path, overwrite=True, save_format="keras")
        return (model, history)

    def save_training_metrics(self, metrics: Dict[str, List[float]], configuration: Configuration, iteration: int) -> None:
        iteration_dir: str = GridSearch.create_dir_if_not_existent(self.get_iteration_dir(configuration, iteration) + "/results")
        with open(iteration_dir + "/training_metrics.csv", 'w') as fp:
            for metric in metrics:
                fp.write(str(metric) + ": " + ", ".join(str(metrics[metric])) + "\n")

    def save_test_metrics(self, metrics: Dict[str, float], configuration: Configuration, iteration: int) -> None:
        iteration_dir: str = GridSearch.create_dir_if_not_existent(self.get_iteration_dir(configuration, iteration) + "/results")
        with open(iteration_dir + "/test_metrics.csv", 'w') as fp:
            for metric in metrics:
                fp.write("{0}: {1}\n".format(metric, str(metrics[metric])))

    def save_averaged_training_metrics(self, metrics: Dict[str, List[float]], configuration: Configuration) -> None:
        conf_results_dir: str = GridSearch.create_dir_if_not_existent(self.get_configuration_dir(configuration) + "/results")
        with open(conf_results_dir + "/avereaged_training_metrics.csv", 'w') as fp:
            for metric in metrics:
                fp.write(str(metric) + ": " + ", ".join(str(metrics[metric])) + "\n")
                pyplot.title("averaged_" + metric)
                pyplot.plot(metrics[metric])
                ax = pyplot.gca()
                ax.set_ylim([0, max(metrics[metric]) * 1.1])
                #if ("val_" + metric) in metrics:
                 #   pyplot.plot(metrics["val_" + metric])
                pyplot.xlabel("epochs")
                pyplot.ylabel("averaged_" + metric, )
                #if ("val_" + metric) in metrics:
                 #   pyplot.plot("averaged_val_" + metric)
                pyplot.savefig(conf_results_dir + "/averaged_" + metric + ".png")
                pyplot.close()

    def save_averaged_test_metrics(self, metrics: Dict[str, float], configuration: Configuration) -> None:
        conf_results_dir: str = GridSearch.create_dir_if_not_existent(self.get_configuration_dir(configuration) + "/results")
        with open(conf_results_dir + "/averaged_test_metrics.csv", 'w') as fp:
            for metric in metrics:
                fp.write("{0}: {1}\n".format(metric, str(metrics[metric])))

    def pause(self) -> None:
        self.do_pause = True

    # Reviewed and Tested
    def is_explored(self, configuration: Configuration) -> bool:
        return configuration.as_tuple() in self.search_db[GridSearch.KEY_EXPLORED_SET]

    def add_to_explored_set(self, configuration: Configuration) -> None:
        self.search_db[GridSearch.KEY_EXPLORED_SET].append(configuration.as_tuple())
        self.search_db.sync()

    def next_configuration(self) -> Configuration:
        if self.get_state() != GridSearch.STATE_VALUE_SEARCHING:
            raise ValueError()
        min_sample_size_2_exp = self.search_db[GridSearch.KEY_MIN_SAMPLE_SIZE_EPX]
        max_sample_size_2_exp = self.search_db[GridSearch.KEY_MAX_SAMPLE_SIZE_EPX]
        layer_descriptor_list = self.search_db[GridSearch.KEY_LAYER_DESCRIPTOR_LIST]
        activation_descriptor_list = self.search_db[GridSearch.KEY_ACTIVATION_DESCRIPTOR_LIST]
        learning_rate_list = self.search_db[GridSearch.KEY_LEARNING_RATE_LIST]
        dropout_rate_list = self.search_db[GridSearch.KEY_DROPOUT_RATE_LIST]
        for sample_size_exp in range(min_sample_size_2_exp, max_sample_size_2_exp + 1):
            for layer_descriptor in layer_descriptor_list:
                for activation_descriptor in activation_descriptor_list:
                    if len(layer_descriptor) != len(activation_descriptor):
                        continue
                    for learning_rate in learning_rate_list:
                        for dropout_rate in dropout_rate_list:
                            yield Configuration(2**sample_size_exp, layer_descriptor, activation_descriptor, learning_rate, dropout_rate)

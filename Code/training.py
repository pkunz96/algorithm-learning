from typing import Tuple, Dict

from itertools import permutations, combinations_with_replacement

from numpy.typing import NDArray

from util.grid_search import GridSearch

from sampling import generate_training_data_record
from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment

label_name_dict = dict()

def sample(size: int) -> Tuple[NDArray, NDArray, Dict[int, str]]:
    x_data, y_data, local_label_name_dict, = generate_training_data_record(straight_insertion_sort, gen_insertion_sort_environment, lambda x: "T_{0}".format(x), size)
    global label_name_dict
    label_name_dict = local_label_name_dict
    return x_data, y_data, label_name_dict

def name_for_label(response: Tuple[int, ...]) -> str:
    return label_name_dict[response[0]]

#Constants

activation_function_list = ["sigmoid", "relu", "tanh"]

# Data

y_data, x_data, label_name_dict = sample(200)

input_dim = y_data.shape[1]
class_count = len(label_name_dict)

# Search Space

training_data_size_exp_2 = 12 # 4096 Samples

min_hidden_layer_count = 1
max_hidden_layer_count = 6

min_layer_size_exp_2 = 5
max_layer_size_exp_2 = 10

layer_descriptor_list = []
activation_descriptor_list = []

for hidden_layer_count in range(min_hidden_layer_count, max_hidden_layer_count + 1):
    # Generates a list of layer descriptors specifying the number of neurons per layer.
    for layer_size_exp_2 in range(min_layer_size_exp_2, max_layer_size_exp_2 + 1):
        layer_size = 2**layer_size_exp_2
        layer_descriptor_list.append([input_dim] + [layer_size] * hidden_layer_count + [class_count])
    local_activation_descriptor_tuple_list = []
    local_activation_descriptor_list = []
    activation_combination_list = map(lambda x: list(x), combinations_with_replacement(activation_function_list, hidden_layer_count + 1))
    for combination in activation_combination_list:
        permutation_list = map(lambda x: list(x), permutations(combination))
        for permutation in permutation_list:
            permutation_tpl = tuple(permutation)
            if permutation_tpl not in local_activation_descriptor_tuple_list:
                local_activation_descriptor_list.append(permutation)
                local_activation_descriptor_tuple_list.append(tuple(permutation))
    local_activation_descriptor_list = list(map(lambda descriptor: descriptor + ["softmax"], local_activation_descriptor_list))
    activation_descriptor_list += local_activation_descriptor_list




grid_search = GridSearch("insertion_sort_62");
grid_search.set_fixed_parameters(input_dim=11, loss_function="sparse_categorical_crossentropy", iterations=3, test_split=0.3, batch_size=256, epochs=1000)
grid_search.set_search_space(training_data_size_exp_2, training_data_size_exp_2, [[input_dim, 100, 100, 100, class_count]], [['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'softmax']], [0.5, 0.1, 0.01, 0.001, 0.0001], [None])

grid_search.search(sample, name_for_label)

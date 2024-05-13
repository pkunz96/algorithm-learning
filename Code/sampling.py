from typing import Callable, List, Tuple, Dict
from math import ceil

import numpy as np
from numpy.typing import NDArray

from algorithms.environment_oop import Environment
from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

problem_mu = 0
problem_sigma = 100

problem_size_mu = 20
problem_size_sigma = 5


# Tested
def generate_problem() -> List[int]:
    problem_size: int = np.random.normal(problem_size_mu, problem_size_sigma, 1)
    return [ceil(value) for value in np.random.normal(problem_mu, problem_sigma, max([ceil(problem_size), 2]))]


# Tested
def sample(sort: Callable[[Environment], List[int]], create_environment: Callable[[List[int]], Environment]) -> NDArray[int]:
    env: Environment = create_environment(generate_problem())
    sort(env)
    return np.array(env.pass_record)


# Tested
def scale(arr: NDArray) -> NDArray:
    arr_flattened = arr.flatten()
    x_min = min(arr_flattened)
    x_max = max(arr_flattened)
    return (arr-x_min)/(x_max-x_min)


# Reviewed
def generate_state_successor_dict(raw_training_record: NDArray[int]) -> Dict[int, List[int]]:
    computation_state_successor_dict: Dict[int, List[int]] = dict()
    for row_index in range(0, len(raw_training_record)-1):
        cur_state = raw_training_record[row_index][-1]
        suc_state = raw_training_record[row_index + 1][-1]
        if cur_state not in computation_state_successor_dict:
            computation_state_successor_dict[cur_state] = []
        successor_list = computation_state_successor_dict[cur_state]
        if suc_state not in successor_list:
            successor_list.append(suc_state)
    return computation_state_successor_dict


#Reviewed
def remap_labels(response_arr: List[List[int]], name_response: Callable[[int], str]) -> Dict[int, str]:
    next_label = 0
    label_name_dict: Dict[int, str] = dict()
    response_label_dict: Dict[int, int] = dict()
    for response in response_arr:
        if response[0] not in response_label_dict:
            response_label_dict[response[0]] = next_label
            label_name_dict[next_label] = name_response(response[0])
            next_label += 1
        response[0] = response_label_dict[response[0]]
    return label_name_dict


def determine_record_size(raw_predictor_arr: NDArray[int], computation_state_successor_dict: Dict[int, List[int]]) -> int:
    count = 0
    for row in raw_predictor_arr:
        state = row[-1]
        if len(computation_state_successor_dict[state]) > 1:
            count += 1
    return count


def generate_training_data_record(sort: Callable[[Environment], List[int]], create_environment: Callable[[List[int]], Environment], name_label: Callable[[int], str], sample_size: int) -> Tuple[NDArray, NDArray, Dict[int, str]]:
    raw_predictor_arr: NDArray[int] = np.array(sample(sort, create_environment))
    computation_state_successor_dict = generate_state_successor_dict(raw_predictor_arr)
    predictor_arr: List[List[int]] = []
    response_arr: List[List[int]] = []
    completed = determine_record_size(raw_predictor_arr, computation_state_successor_dict) >= sample_size
    while not completed:
        raw_predictor_arr = np.append(raw_predictor_arr, sample(sort, create_environment), axis=0)
        computation_state_successor_dict = generate_state_successor_dict(raw_predictor_arr)
        completed = determine_record_size(raw_predictor_arr, computation_state_successor_dict) >= sample_size
    for step_index in range(0, len(raw_predictor_arr) - 1):
        cur_predictor_row = raw_predictor_arr[step_index]
        cur_state = cur_predictor_row[-1]
        if len(computation_state_successor_dict[cur_state]) > 1:
            predictor_arr.append(cur_predictor_row.tolist())
            suc_predictor_row = raw_predictor_arr[step_index + 1]
            suc_state = suc_predictor_row[-1]
            response_arr.append([suc_state])
        if len(predictor_arr) == sample_size:
            break
    predictor_np_arr = np.array(predictor_arr)
    predictor_np_arr_left = predictor_np_arr[:,:-1]
    predictor_np_arr_left = scale(predictor_np_arr_left) # Scaled Predictor Array
    predictor_np_arr_right = predictor_np_arr[:,-1:]
    ohe = OneHotEncoder()
    predictor_np_arr_right = ohe.fit_transform(predictor_np_arr_right).toarray()
    label_name_dict = remap_labels(response_arr, name_label)
    return np.concatenate((predictor_np_arr_left, predictor_np_arr_right), axis=1), np.array(response_arr), label_name_dict


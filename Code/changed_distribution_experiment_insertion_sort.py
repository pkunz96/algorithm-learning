from typing import Tuple, Dict
from numpy.typing import NDArray
from numpy import save
import tensorflow
from changed_sampling import generate_training_data_record
from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os;

def sample(size: int, mu: int, sigma: int) -> Tuple[NDArray, NDArray, Dict[int, str]]:
    x_data, y_data, local_label_name_dict, = generate_training_data_record(straight_insertion_sort, gen_insertion_sort_environment, lambda x: "T_{0}".format(x), size, mu, sigma)
    global label_name_dict
    label_name_dict = local_label_name_dict
    return x_data, y_data, label_name_dict

#init_mu = -50
#init_sigma = 0

"""
with open(experiment_name + ".txt", 'w') as fp:
    for mu_counter in range(0, 10):
        for sigma_counter in range(0, 5):
            loss = 0;
            accuracy = 0;
            mu = init_mu + mu_counter * 10
            sigma = init_sigma + sigma_counter * 10
            iteration_count = 5
            for iteration in range(0, iteration_count):
                x_data, y_data, label_name_dict = sample(2 ** sample_size_exp_2, mu, sigma)
                eval_results = model.evaluate(x_data, y_data, verbose=0)
                loss += eval_results[0]
                accuracy += eval_results[1]
            loss /= iteration_count
            accuracy /= iteration_count
            fp.write("mu: {0} sigma: {1} loss: {2} accuracy: {3}\n".format(mu, sigma, loss, accuracy))
            fp.flush()
"""

sample_size_exp_2 = 14
experiment_name = "changed_distribution_experiment_insertion_sort_changed_mu_0_sigma_150"

model = load_model(input("Model Path?"))


mu = 0;
sigma = 150;

#problem_mu = 0
#problem_sigma = 100

def create_dir_if_not_existent(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path

create_dir_if_not_existent("./" + experiment_name)

with open("./" + experiment_name + "/results.txt", 'w') as fp:
    loss = 0;
    accuracy = 0;
    iteration_count = 5
    for iteration in range(0, iteration_count):
        create_dir_if_not_existent("./" + experiment_name + "/" + str(iteration))
        x_data, y_data, label_name_dict = sample(2 ** sample_size_exp_2, mu, sigma)
        save("./" + experiment_name + "/" + str(iteration) + "/x_data" , x_data)
        save("./" + experiment_name + "/" + str(iteration) + "/y_data", y_data)
        eval_results = model.evaluate(x_data, y_data, verbose=0)
        loss += eval_results[0]
        accuracy += eval_results[1]
    loss /= iteration_count
    accuracy /= iteration_count
    fp.write("mu: {0} sigma: {1} loss: {2} accuracy: {3}\n".format(mu, sigma, loss, accuracy))
    fp.flush()

print(eval_results)

import os
import numpy as np

def calculate_meaen_std(data_root):
    samples = []
    for classname in os.listdir(data_root):
        class_folder = os.path.join(data_root, classname)

        for sample in os.listdir(class_folder):
            sample_filepath = os.path.join(class_folder, sample)

            with open(sample_filepath, 'r') as sample_file:
                for value in sample_file.read().split('\n'):
                    if value != '':
                        samples.append(int(value))
    return [np.mean(samples), np.std(samples)]

#print(calculate_meaen_std('./dataset'))

#[953.4888085135459, 105.17745351508788]

def calculate_min_max_value(data_root):
    min_value = 999999
    max_value = 0
    samples = []
    for classname in os.listdir(data_root):
        class_folder = os.path.join(data_root, classname)

        for sample in os.listdir(class_folder):
            sample_filepath = os.path.join(class_folder, sample)

            with open(sample_filepath, 'r') as sample_file:
                for value in sample_file.read().split('\n'):
                    if value != '':
                        if min_value > int(value):
                            min_value = int(value)
                        if max_value < int(value):
                            max_value = int(value)
                        #print(min_value, max_value)
    print(min_value)
    return [min_value, max_value]

print(calculate_min_max_value('./dataset'))
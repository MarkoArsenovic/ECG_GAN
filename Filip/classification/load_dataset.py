from os import listdir
from os.path import isfile, join
from config import confuguration
import matplotlib.pyplot as plt
import csv

def load_train_test_ECG_dateset():
    list_files = [f for f in listdir(confuguration.db_path) if isfile(join(confuguration.db_path, f))]

    records = []
    annotations = []

    for f in list_files:
        if f[-3:] == 'csv':
            records.append(f)
        if f[-3:] == 'txt':
            annotations.append(f)

    records.sort()
    annotations.sort()

    x = []
    y = []

    limit_class_N = 0

    for r in range(len(records)):
        signals = []
        print(confuguration.db_path + records[r])
        with open(confuguration.db_path + records[r], 'r') as record_file:
            spamreader = list(csv.reader(record_file, delimiter=',', quotechar='|'))
            for row in range(1, len(spamreader)):
                signals.insert(row, int((spamreader[row])[1]))
                
        """
        if r == 0:
            plt.title("Record")
            plt.plot(signals)
            plt.show()
        """
        
        with open(confuguration.db_path + annotations[r], 'r') as annotation_file:
            annotation_lines = list(annotation_file.readlines())
            for row in range(1, len(annotation_lines)):
                row_elem = annotation_lines[row].split(' ')
                row_elem = list(filter(lambda a: a != '', row_elem))
                for class_index in range(len(confuguration.classes)):
                    position_of_class = int(row_elem[1])
                    if confuguration.classes[class_index] == row_elem[2] and (position_of_class - confuguration.kernel_size) >= 0 and (position_of_class + confuguration.kernel_size) <= len(signals):
                        if class_index == 0:
                            limit_class_N += 1
                            if limit_class_N < confuguration.limit_class_N:
                                x.append(signals[position_of_class - confuguration.kernel_size : position_of_class + confuguration.kernel_size])
                                y.append(class_index)
                        else:
                            x.append(signals[position_of_class - confuguration.kernel_size : position_of_class + confuguration.kernel_size])
                            y.append(class_index)
                        break
            print(len(x))
        """            
        if r == 0:
            plt.title("One sample")
            plt.plot(x[0])
            plt.show()
        """
    return [x, y]
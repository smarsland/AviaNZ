"""
6/10/2022
Checked: 28/10/2022
Author: Virginia Listanti

This script prepare a .csv for each class with the list of the train files. After that saves a .csv with a list of the
test dataset
"""

import os
import csv
import random
import numpy as np

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
            "Smaller_Dataset2\\Original"
save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
           "Smaller_Dataset2"

labels_lists = ["D", "E", "J", "K", "L", "M", "O", "R", "Z"]
# num_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# random.shuffle(num_list)
# id_list = num_list[:7]
fieldnames1 = ['Syllable']

test_set = []
for label in labels_lists:
    num_list =[]
    for file in os.listdir(directory):
        if file.endswith('.csv') and file[0] == label:
            num_list.append(file[1:-7])
    random.shuffle(num_list)
    id_list = num_list[:int(np.floor(len(num_list)*2/3))]

    train_set = []
    for id in num_list:
        if id in id_list:
            train_set.append(label+id+'_IF.csv')
        else:
            test_set.append(label + id + '_IF.csv')

    save_name1 = "Class_"+label+".csv"
    csv_path1 = save_dir + "\\" + save_name1
    with open(csv_path1, 'w', newline='') as csvfile:
        Writer = csv.DictWriter(csvfile, fieldnames=fieldnames1)
        Writer.writeheader()
    #
    for syllable in train_set:
        with open(csv_path1, 'a', newline='') as csvfile:  # should be ok
            Writer = csv.DictWriter(csvfile, fieldnames=fieldnames1)
            Writer.writerow({"Syllable": syllable})

    del train_set

save_name2 = "Test_file_list.csv"
csv_path2 = save_dir + "\\" + save_name2
fieldnames2 = ["Syllable", "Label"]
with open(csv_path2, 'w', newline='') as csvfile:
    Writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
    Writer.writeheader()
#
for syllable in test_set:
    with open(csv_path2, 'a', newline='') as csvfile:  # should be ok
        Writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
        Writer.writerow({"Syllable": syllable, "Label": syllable[0]})
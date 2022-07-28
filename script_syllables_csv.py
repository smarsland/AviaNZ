"""
26/07/2022
Author: Virginia Listanti

This script create a .csv file to check kiwi syllables
"""

import os
import csv

extracted_fold = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\" \
                 "Kiwi_IndividualID\\extracted"

fieldnames = ["Syllable Name", "Tag"]

save_path_csv ="C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
               "syllable_check.csv"

with open(save_path_csv, 'w', newline='') as csvfile:
    Writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    Writer.writeheader()

for file in os.listdir(extracted_fold):
    if file.endswith('.wav'):
        with open(save_path_csv, 'a', newline='') as csvfile:  # should be ok
            Writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            Writer.writerow({"Syllable Name": file[:-4], "Tag":' '})
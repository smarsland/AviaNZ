"""
16/08/2022
Author: Virginia Listanti

This script reads the distance matrix and find the 3 most similar syllables and save them in .csv file
"""
import os
import csv
import numpy as np

results_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
                   "\\exemplars\\Models\\Distance_ matrices\\Exemplars_vs_Models\\Test4"

original_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\" \
                     "Kiwi_IndividualID\\exemplars\\Models\\Exemplars_Ridges"

fake_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
                 "\\exemplars\\Models\\Models_Ridges"

#create list of syllables
list_syllables =[]
for file in os.listdir(original_directory):
    if file.endswith('_IF.csv'):
        list_syllables.append(file[:-7])

#create list of fake syllables
fake_syllables = []
for file in os.listdir(fake_directory):
    if file.endswith('_IF.csv'):
        fake_syllables.append(file[:-7])

fieldnames = ['Syllable', 'First', 'Second', 'Third']

## GEODESIC MATRIX ##
geo_path = results_directory + "\\" + "Geodesic.txt"
# read matrix
G_dist = np.loadtxt(geo_path)
#open csv file
csvfilename = results_directory + "\\" + "Geodesic_similar_syllables.csv"
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    #read best matches and store them in csv file
    for i in range(len(list_syllables)):
        distances = G_dist[i, :]
        indices = np.argsort(distances)[:3]
        writer.writerow({'Syllable': list_syllables[i], 'First': fake_syllables[indices[0]],
                         'Second': fake_syllables[indices[1]],  'Third': fake_syllables[indices[2]]})

## CROSSCORRELATION MATRIX ##
crosscorr_path = results_directory + "\\" + "cross-correlation.txt"
# read matrix
Cross_dist = np.loadtxt(crosscorr_path)
#open csv file
csvfilename = results_directory + "\\" + "Crosscorr_similar_syllables.csv"
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    #read best matches and store them in csv file
    for i in range(len(list_syllables)):
        distances = Cross_dist[i, :]
        indices = np.argsort(distances)[:3]
        writer.writerow({'Syllable': list_syllables[i], 'First': fake_syllables[indices[0]],
                         'Second': fake_syllables[indices[1]],  'Third': fake_syllables[indices[2]]})
## DTW matrix ##
dtw_path = results_directory + "\\" + "DTW.txt"
# read matrix
DTW_dist = np.loadtxt(dtw_path)
#open csv file
csvfilename = results_directory + "\\" + "DTW_similar_syllables.csv"
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    #read best matches and store them in csv file
    for i in range(len(list_syllables)):
        distances = DTW_dist[i, :]
        indices = np.argsort(distances)[:3]
        writer.writerow({'Syllable': list_syllables[i], 'First': fake_syllables[indices[0]],
                         'Second': fake_syllables[indices[1]],  'Third': fake_syllables[indices[2]]})

## SSD MATRIX ##
ssd_path = results_directory + "\\" + "SSD.txt"
# read matrix
SSD_dist = np.loadtxt(ssd_path)
#open csv file
csvfilename = results_directory + "\\" + "SSD_similar_syllables.csv"
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    #read best matches and store them in csv file
    for i in range(len(list_syllables)):
        distances = SSD_dist[i, :]
        indices = np.argsort(distances)[:3]
        writer.writerow({'Syllable': list_syllables[i], 'First': fake_syllables[indices[0]],
                         'Second': fake_syllables[indices[1]],  'Third': fake_syllables[indices[2]]})

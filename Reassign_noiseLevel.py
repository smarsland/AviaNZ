"""
8/04/2022
Author: Virginia Listanti

This script purpose is to rearrange the noise levels
"""

# import SignalProc
# import IF as IFreq
import numpy as np
# import matplotlib.pyplot as plt
import os
import csv
# import ast
import shutil


def noise_mapping(signal_id):
    """
    This script gives as an output a list with the true noise level
    """

    mapping = []

    if signal_id == 'pure_tone':
        mapping = [str(2), str(4), str(14), str(3), str(11), str(13), str(12), str(1), str(8), str(10), str(6), str(5),
                   str(9), str(7)]

    elif signal_id == 'linear_upchirp':
        mapping = [str(7), str(3), str(12), str(4), str(14), str(5), str(9), str(1), str(8), str(10), str(6), str(13),
                   str(11), str(2)]

    elif signal_id == 'linear_downchirp':
        mapping = [str(1), str(8), str(9), str(5), str(2), str(7), str(12), str(6), str(13), str(3), str(4), str(10),
                   str(11), str(14)]

    elif signal_id == 'exponential_downchirp':
        mapping = [str(1), str(8), str(6), str(10), str(14), str(12), str(4), str(11), str(2), str(7), str(9), str(13),
                   str(5), str(3)]

    elif signal_id == 'exponential_upchirp':
        mapping = [str(8), str(13), str(12), str(9), str(2), str(7), str(1), str(10), str(11), str(4), str(14), str(5),
                   str(3), str(6)]

    return mapping
def sort_test_list(s):
    "This script sort Test_list using nubers"

    test_num=[]
    for element in s:
        test_num.append(element[5:])

    test_num.sort(key=int)

    new_list =[]

    for element in test_num:
        new_list.append('Test_'+element)
    return new_list

def sort_noise_list(s):
    "This script sort Test_list using nubers"

    test_num = []
    for element in s:
        test_num.append(element[12:-4])

    test_num.sort(key=int)

    new_list =[]

    for element in test_num:
        new_list.append('noise_level_'+element+'.csv')
    return new_list

def rename_noise_level_files(workdir, signal_id):
    """
    this function rename the noise_level file
    """

    # recover noise level file list
    file_list = os.listdir(workdir)
    noise_level_list = []
    for element in file_list:
        if element.startswith('noise_level_'):
            noise_level_list.append(element)

    noise_level_list = sort_noise_list(noise_level_list)

    noise_map=noise_mapping(signal_id)
    new_noise_level_list = []
    for element in noise_level_list:
        #copy with a new name
        new_noise_level_list.append(element[:11] + '_' + noise_map[noise_level_list.index(element)] + '_new.csv')
        path1 = workdir + '\\' + element
        path2 = workdir + '\\' + element[:11] + '_' + noise_map[noise_level_list.index(element)] + '_new.csv'
        shutil.copy(path1, path2)
        os.remove(path1)
        del path1, path2

    for element in new_noise_level_list:
        #rename new files
        path1 = workdir + '\\' + element
        path2 = path1[:-8] + '.csv'
        os.rename(path1, path2)
        del path1, path2

    return


def rearrange_metrics_files(workdir, signal_id):

    """
    This function rearrange the columns of the metrics files
    """
    # recover noise level file list
    file_list = os.listdir(workdir)
    # recover noise level file list
    metrics_list = []
    for element in file_list:
        if element.startswith('noise_levels_'):
            metrics_list.append(element)

    #column map
    column_map = noise_mapping(signal_id)

    fieldnames = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8', 'Level 9',
                  'Level 10', 'Level 11', 'Level 12', 'Level 13', 'Level 14']

    for file in file_list:
        print('Rearranging file ', file)
        # read .csv file
        csvfilename = workdir+ '\\'+file
        old_matrix = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                old_matrix.append(row)

        # remove old file
        os.remove(csvfilename)
        old_matrix  = old_matrix[1:][:]
        #define new matrix
        [m, n] = np.shape(old_matrix)
        new_matrix = np.zeros((m, n))
        old_matrix = np.array(old_matrix)

        #rearrange
        for k in range(n):
            new_matrix[:,int(column_map[k])-1] = old_matrix[:,k]

        #save new matrix
        with open(csvfilename, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for h in range(m):
                row = {}
                for j in range(len(fieldnames)):
                    row[fieldnames[j]] = new_matrix[h, j]
                writer.writerow(row)

        del old_matrix, new_matrix

    return




# signal type we are analysing
signal_id='pure_tone'
# signal_id = 'linear_upchirp'
#signal_id = 'linear_downchirp'
# signal_id = 'exponential_upchirp'
# signal_id = 'exponential_downchirp'

test_result_dir = "C:\\Users\\Virginia\\Documents\Work\\IF_extraction\\Test_Results"

test_list = os.listdir(test_result_dir)

test_list = sort_test_list(test_list)

for test_folder in test_list:
    work_dir = test_result_dir + '\\' + test_folder + '\\' + signal_id

    if not os.path.exists(work_dir):
        continue



    print('Rearranging ', test_folder)
    #if test_folder!='Test_0':
    rename_noise_level_files(work_dir, signal_id)

    rearrange_metrics_files(work_dir, signal_id)


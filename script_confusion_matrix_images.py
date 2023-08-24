"""
18/11/2022
Author: Virginia Listanti

This script produces heatmap images of confusion matrices for chapter 5 of my thesis
"""

#lybraries
import numpy as np
import matplotlib.pyplot as plt
import csv


# Function create and save heat map
def heatmap(A, lab, savefile):
    """
    This function create an heatmap of the matrix A with labels lab and save it save file
    """

    fig, ax = plt.subplots()
    # im = ax.imshow(A, cmap='Wistia')
    im = ax.imshow(A, cmap='YlOrRd')
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(lab)), labels=lab)
    ax.set_yticks(np.arange(len(lab)), labels=lab)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(lab)):
        for j in range(len(lab)):
            if A[i,j] < 50:
                col = "black"
            else:
                col = "white"

            text = ax.text(j, i, A[i, j],
                           ha="center", va="center", color=col)

    #     ax.set_title("Confusion matri")
    fig.tight_layout()
    plt.savefig(savefile, dpi=200)

    return


# label_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#              "Standard_Linear\\Test_3\\Labels_comparison_df.csv"
# label_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#              "Standard_Mel\\Test_127\\Labels_comparison_combine3.csv"
# label_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#              "Reassigned_Linear\\Test_209\\Labels_comparison_combine2_df.csv"
# label_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#              "Reassigned_Mel\\Test_837\\Labels_comparison_pure.csv"
# label_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#              "Multitapered_Linear\\Test_627\\Labels_comparison_combine3.csv
# label_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#              "Multitapered_Mel\\Test_1001\\Labels_comparison_df.csv"
label_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
             "Thesys_Review_April_2023\\Reassigned_Linear\\Test_1059\\Labels_comparison_combine2.csv"
# save_fold = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter 5"
save_fold = "C:\\Users\\Virginia\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
            "Thesys_Review_April_2023\\Confusion_matrix\\Test_1059"
# save_file = "Multitapered_mel_conf_matrix.jpg"
save_file = "Reassign_linear_conf_matrix.jpg"
save_path = save_fold + "\\" +save_file
# label_list = ['D', 'E', 'J', 'K', 'L', 'M','O', 'R', 'Z']
label_list = ['D', 'E', 'J', 'K', 'L', 'M','O', 'Z']
field = []
rows = []
with open(label_file, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

N = len(rows)
print(N)

# recover true and assigned label list
true_labels_list = []
assigned_labels_list = []
for i in range(N):
    true_labels_list.append(rows[i][1])
    assigned_labels_list.append(rows[i][6])

# print(true_labels_list)
# print(assigned_labels_list)

# count label elements
number_label = {}
for label in true_labels_list:
    if label in number_label:
        number_label[label]+= 1
    else:
        number_label[label] = 1
print(number_label)

#confusion matrix: rows assigned labels columns real labels

confusion_matrix = np.zeros((8,8))

for i in range(N):
    true_index = label_list.index(true_labels_list[i])
    assigned_index = label_list.index(assigned_labels_list[i])
    confusion_matrix[assigned_index, true_index] += 1

confusion_matrix[:,0]/=3
confusion_matrix[:,1]/=4
confusion_matrix[:, 2]/=4
confusion_matrix[:, 3]/=4
confusion_matrix[:, 4]/=5
confusion_matrix[:, 5]/=4
confusion_matrix[:, 6]/=5
confusion_matrix[:, 7]/=4
# confusion_matrix[:,8]/=5

confusion_matrix*=100

confusion_matrix = np.round(confusion_matrix, decimals =2)

heatmap(confusion_matrix, label_list, save_path)

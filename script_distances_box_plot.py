"""
18/11/2022
Author: Virginia Listanti

This script produces box-plot like images of the distances along all the tfr tests
"""

#lybraries
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def read_csv_table(file_path):
    " This function read a .csv table"

    rows = []
    with open(file_path, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    return rows

# analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#                "Standard_Linear"
# analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\Standard_Mel"
# analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#                "Reassigned_Linear"
# analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#                "Reassigned_Mel"
# analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
#                "Multitapered_Linear"
analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_syllable_TFR_experiment\\" \
               "Multitapered_Mel"
# save_file = 'standard_linear_metrics_all.jpg'
# save_file2 = 'standard_linear_metrics_all2.jpg'
# col = 'purple'
# save_file = 'standard_mel_metrics_all.jpg'
# save_file2 = 'standard_mel_metrics_all2.jpg'
# col = 'mediumvioletred'
# save_file = 'Reassigned_linear_metrics_all.jpg'
# save_file2 = 'Reassigned_linear_metrics_all2.jpg'
# col = 'forestgreen'
# save_file = 'Reassigned_mel_metrics_all.jpg'
# save_file2 = 'Reassigned_mel_metrics_all2.jpg'
# col = 'lightgreen'
# save_file = 'Multitapered_linear_metrics_all.jpg'
# save_file2 = 'Multitapered_linear_metrics_all2.jpg'
# col = 'orangered'
save_file = 'Multitapered_mel_metrics_all.jpg'
save_file2 = 'Multitapered_mel_metrics_all2.jpg'
col = 'orange'

save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter 5"
save_path = save_dir + "\\" +save_file
save_path2 = save_dir + "\\" +save_file2
test_lim = 627

M1 = []
M2 = []
M3 = []
M4 = []
M5 = []
M6 = []
M7 = []
M8 = []
M9 = []
M10 = []
M11 = []
M12 = []
M13 = []
M14 = []
M15 = []
M16 = []
M17 = []
M18 = []
M19 = []
M20 = []
M21 = []
M22 = []
M23 = []
M24 = []
M25 = []
M26 = []
M27 = []
M28 = []

metrics_file = "Accuracy_results.txt"

for fold in os.listdir(analysis_dir):
    Test_id = int(fold[5:])
    # if Test_id > test_lim:
    #     continue

    print(fold)
    analysis_fold = analysis_dir + '\\' + fold
    file_path = analysis_fold + '\\' + metrics_file

    with open(file_path) as f:
        lines = f.readlines()

    # base
    M1.append(float(lines[2][15:-1]))
    M2.append(float(lines[3][15:-1]))
    M3.append(float(lines[4][15:-1]))
    M4.append(float(lines[5][20:-1]))

    #base +df
    M5.append(float(lines[7][20:-1]))
    M6.append(float(lines[8][20:-1]))
    M7.append(float(lines[9][20:-1]))
    M8.append(float(lines[10][24:-1]))

    #combination 2 base
    M9.append(float(lines[12][21:-1]))
    M10.append(float(lines[13][21:-1]))
    M11.append(float(lines[14][21:-1]))
    M12.append(float(lines[15][21:-1]))
    M13.append(float(lines[16][21:-1]))
    M14.append(float(lines[17][21:-1]))

    # combination 2 base +df
    M15.append(float(lines[19][30:-1]))
    M16.append(float(lines[20][30:-1]))
    M17.append(float(lines[21][30:-1]))
    M18.append(float(lines[22][30:-1]))
    M19.append(float(lines[23][30:-1]))
    M20.append(float(lines[24][30:-1]))

    # combination 3 base
    M21.append(float(lines[26][27:-1]))
    M22.append(float(lines[27][27:-1]))
    M23.append(float(lines[28][27:-1]))
    M24.append(float(lines[29][27:-1]))

    # combination 3 base +df
    M25.append(float(lines[31][36:-1]))
    M26.append(float(lines[32][36:-1]))
    M27.append(float(lines[33][36:-1]))
    M28.append(float(lines[34][36:-1]))

M1 = np.array(M1) *100
M2 = np.array(M2) *100
M3 = np.array(M3) *100
M4 = np.array(M4) *100
M5 = np.array(M5) *100
M6 = np.array(M6) *100
M7 = np.array(M7) *100
M8 = np.array(M8) *100
M9 = np.array(M9) *100
M10 = np.array(M10) *100
M11 = np.array(M11) *100
M12 = np.array(M12) *100
M13 = np.array(M13) *100
M14 = np.array(M14) *100
M15 = np.array(M15) *100
M16 = np.array(M16) *100
M17 = np.array(M17) *100
M18 = np.array(M18) *100
M19 = np.array(M19) *100
M20 = np.array(M20) *100
M21 = np.array(M21) *100
M22 = np.array(M22) *100
M23 = np.array(M23) *100
M24 = np.array(M24) *100
M25 = np.array(M25) *100
M26 = np.array(M26) *100
M27 = np.array(M27) *100
M28 = np.array(M28) *100
# D = M1
D = np.vstack((M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20, M21, M22, M23,
               M24, M25, M26, M27, M28))
labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17',
          'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M26', 'M27', 'M28']
# plot:
fig, ax = plt.subplots()

pos = np.arange(2, 113, 4)
ax.eventplot(D, orientation="vertical", lineoffsets=pos, linelength=2,  linewidth=1, color = col )

ax.set(xlim=(0, 114), xticks=np.arange(2, 113, 4),
       ylim=(0, 100), yticks=np.arange(0, 110, 10))

ax.set_xticklabels(labels, size = 5 )

# plt.show()
# fig.tight_layout()
plt.savefig(save_path, dpi=200)

fig, ax = plt.subplots()

pos = np.arange(2, 113, 4)
ax.boxplot(D.T, positions=pos,  widths=2,
           showmeans=False, showfliers=False, patch_artist=True,
           medianprops={"color": "white", "linewidth": 1},
           boxprops={"facecolor": col, "edgecolor": "white",
                     "linewidth": 0.5},
           whiskerprops={"color": col, "linewidth": 1.5},
           capprops={"color": col, "linewidth": 1.5}
           )

ax.set(xlim=(0, 114), xticks=np.arange(2, 113, 4),
       ylim=(0, 100), yticks=np.arange(0, 110, 10))

ax.set_xticklabels(labels, size = 5 )

# plt.show()
fig.tight_layout()
plt.savefig(save_path2, dpi=200)





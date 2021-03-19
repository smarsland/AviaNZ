## script to copy recursevely file from shareharddisk to local hardisk

import os
import shutil
import csv


remote_dir="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Bittern_Harry"
local_dir="/run/media/listanvirg/LaCie/Sound Files"

#define csv for log files not copyed
fieldnames=['Directory_tree', 'File_name']
csvfilename="/home/listanvirg/log_nocopy.csv"

with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

#print(len(remote_dir))
for roots, dirs, files in os.walk(remote_dir):
    for file in files:
        if file.endswith(".data"):
            remote_file_path=os.path.join(roots, file)
            #print(remote_file_path)
            dir_tree=remote_file_path[67:-25]
            #print(dir_tree)
            local_file_path=local_dir+"/"+dir_tree+"/"+file
            local_wavfile_path=local_file_path[:-5]
            #print(local_file_path)
            if os.path.isfile(local_wavfile_path):
                if os.path.isfile(local_file_path):
                    print('Data file already present')
                    with open(csvfilename,'a', newline='') as csvfile:
                        writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({'Directory_tree':dir_tree, 'File_name':file })
                else:
                    shutil.copy(remote_file_path,local_file_path)
                    print("File ", file, " in ", dir_tree, " copied to ", local_file_path)



#check if all .data files copyied
#define second log file

#define csv for log files not copyed
fieldnames=['Directory_tree', 'File_name']
csvfilename2="/home/listanvirg/log_nodata.csv"

with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for roots, dirs, files in os.walk(local_dir):
    for file in files:
        if file.endswith(".wav") and not os.path.isfile(file+'.data'):
            filedata=file+'.data'
            local_file_path = os.path.join(roots, filedata)
            dir_tree = local_file_path[len(local_dir)+1:-25]
            print('\n\n Data file missing')
            print("\n File ", filedata, " in ", dir_tree)

            with open(csvfilename2, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Directory_tree': dir_tree, 'File_name': filedata})

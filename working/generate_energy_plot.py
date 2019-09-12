#generate_energy_plot

#Virginia Listanti

#This script generate energy plot of the Jinnai energy for the relevant nodes for the filter

import os
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

for root, dirs, files in os.walk('/home/listanvirg/Filter test/Second slot/Prelimary test/'):
    for file in files:
        if file.endswith('energy.txt') and os.stat(root + '/' + file).st_size != 0:
            #Read energy file
            with open(str(root+'/'+file)) as f:
                energy = [map(float, row) for row in csv.reader(f)]
            fig_name=root+'/'+file[:-4]+'.jpg'
            #plot energy
            fig, axes = plt.subplots(2,2,sharex=True, sharey=True)
            fig.set_title('Energy plot'+str(file[:-11]))
            axes[0,0].plot(energy[0,:])
            axes[0,0].set_title('Node 35')
            axes[0, 1].plot(energy[1, :])
            axes[0, 1].set_title('Node 43')
            axes[1, 0].plot(energy[2, :])
            axes[1, 0].set_title('Node 36')
            axes[1, 1].plot(energy[3, :])
            axes[1, 1].set_title('Node 45')
            #save energy plot
            plt.savefig(fig_name)
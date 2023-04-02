"""
27/11/2022
Author: Virginia Listanti

This script plot IF curve os synthtic kiwi syllables
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter6"

syllable_ID = 'Z'
syllable_IF_path = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\Models\\" \
                   + syllable_ID + "_fake_IF_harmonic_1.csv"

list_freq_labels = ['0', '1000', '2000', '3000', '4000']
curve = np.loadtxt(syllable_IF_path, skiprows=1, delimiter=',')
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(curve)
ax.set_ylim(0, 4000)
ax.yaxis.set_major_locator(LinearLocator(5))
# ax[i,j].yaxis.set_ticks(np.arange(17))
ax.set_yticklabels(list_freq_labels, size = 8)
ax.set_ylabel('Frequency (Hz)', size = 10)
ax.set_xlabel('Time (samples)', size = 10)
fig_name = save_dir + "\\class"+syllable_ID+"_IFcurve.jpg"
fig.tight_layout()
plt.savefig(fig_name, dpi = 200)
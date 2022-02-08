#29/10/2021
# Author: Virginia Listanti
#help script for TF tests

import SignalProc
import IF2 as IFreq
#using IF2 just for safety
import numpy as np
# from scipy.io.wavfile import write
import wavio
import matplotlib.pyplot as plt
import WaveletFunctions

#if multiple files, can just loop over using f-strings
# for n in range(0,15):
#     if n <= 9:
#         test_name=f"pure_tone_0{n}"
#         file_name=f"C:\\Users\\Harvey\\Documents\\GitHub\\AviaNZ\\Toy signals\\pure_tone\\pure_tone_0{n}.wav"
#     else:
#         test_name = f"pure_tone_{n}"
#         file_name = f"C:\\Users\\Harvey\\Documents\\GitHub\\AviaNZ\\Toy signals\\pure_tone\\pure_tone_{n}.wav"
test_name = "Exponential_downchirp_3d"
file_name = "C:\\Users\\Harvey\\Documents\\GitHub\\AviaNZ\\Toy signals\\Exponential_downchirp\\Exponential_downchirp_00.wav"
#parameters
window = 0.25
inc = 0.256

#calling IF class
IF = IFreq.IF(method=2, pars=[0, 1])

#calling signal proc
sp = SignalProc.SignalProc(window, inc)
sp.readWav(file_name)
fs = sp.sampleRate

#calling wavelet functions
wf = WaveletFunctions.WaveletFunctions(sp.data,'dmey2',None,fs)

# number of samples in window
win_sr = int(np.ceil(window * fs))
# number of sample in increment
inc_sr = int(np.ceil(inc * fs))
#level of tree
wf.maxLevel = 5
# output columns dimension equal to number of sliding window
N = int(np.ceil(len(sp.data) / inc_sr))
coefs = np.zeros((2 ** (wf.maxLevel + 1) - 2, N))
allnodes = range(2 ** (wf.maxLevel + 1) - 1)

#wavelet packet decomposition
wf.WaveletPacket(allnodes, mode='symmetric', antialias=True, antialiasFilter=True)

#evaluate scalogram
for node in allnodes:
    nodeE, noderealwindow = wf.extractE(node, window, wpantialias=True)
    # fixing in case window differs slightly. copied from wavelet segments
    if N == len(nodeE) + 1:
        coefs[node - 1, :-1] = nodeE
        coefs[node - 1, -1] = coefs[node - 1, -2]
        # repeat last element
    elif N == len(nodeE) - 1:
        coefs[node - 1, :] = nodeE[:-1]
        # drop last WC
    elif np.abs(N - len(nodeE)) > 1:
        print("ERROR: lengths of annotations and energies differ:", N, len(nodeE))
        #if too different, just print error
        print(nodeE)
    else:
        coefs[node - 1, :] = nodeE

print("Scalogram Dim =", np.shape(coefs))

#Standard freq ax
fstep = (fs / 2) / np.shape(coefs)[0]
freqarr =np.arange(fstep, fs / 2 + fstep, fstep)

#setting parametes for ecurve
wopt = [fs, window]
#calling ecurve
tfsupp,_,_=IF.ecurve(coefs,freqarr,wopt)

#setting up lists for iteration
TFR = []
TFR.append(coefs.copy())
ridgepaint = []
ridgepaint.append(np.zeros_like(coefs))
tfsupplist = []
tfsupplist.append(tfsupp.copy())

#finding the next components, arbitrary range right now.
for cn in range(0,5):
    if np.mean(TFR[cn]) <= 0:
        #this should be stopping it searching for components once there're none left. It is wrong right now, not sure why.
        break
    else:
        del IF
        IF = IFreq.IF(method=2, pars=[0, 1])
        #resetting IF else it doesn't like to update. Probably could fix
        TFR.append(coefs.copy())
        ridgepaint.append(np.zeros_like(coefs))
        for i in range(np.shape(coefs)[1]):
            for n in range(int(np.floor(np.amin((tfsupplist[cn][0,i])/fstep)-.5)),int(np.ceil(np.amax((tfsupplist[cn][0,i])/fstep))+.5)):
                TFR[cn][n,i] = 0
                ridgepaint[cn][n,i] = coefs[n,i]
        TFR.append(TFR[cn].copy())
        ridgepaint.append(ridgepaint[cn].copy())
        tfsupp,_,_=IF.ecurve(TFR[cn],freqarr,wopt)
        tfsupplist.append(tfsupp.copy())

data = wf.tree[0]
new_signal = np.zeros(len(data))
for node in range(1,2 ** (wf.maxLevel + 1) - 1):
    for i in range(len(wf.tree[node])):
        if ridgepaint[4][node-1,int(np.floor(i*20/len(wf.tree[node])))] == 0:
            wf.tree[node][i] = 0
    tmp = wf.reconstructWP2(node)[0:len(data)]
    new_signal = new_signal + tmp

wavio.write("ridgepainttest2.wav", new_signal, 16000,sampwidth=2)

# write("writetest.wav",16000, new_signal.astype(np.int16))

#save picture
# change fig_name with the path you want
# fig_name="C:\\Users\\Harvey\\Desktop\\Uni\\avianz\\test plots"+"\\"+test_name+".jpg"
# plt.rcParams["figure.autolayout"] = True
# fig, ax = plt.subplots(1, 3, sharex=True)
# ax[0].imshow(np.flipud(coefs), extent=[0, np.shape(coefs)[1], 0, np.shape(coefs)[0]], aspect='auto')
# x = np.array(range(np.shape(coefs)[1]))
# ax[1].imshow(np.flipud(ridgepaint[4]), extent=[0, np.shape(coefs)[1], 0, np.shape(coefs)[0]], aspect='auto')
# #plotting the ridge curves
# for i in range(len(tfsupplist)):
#     ax[0].plot(x, tfsupplist[i][0, :] / fstep, linewidth=1, color='r')
#     ax[2].plot(x, tfsupplist[i][0, :], color='green')
# ax[2].set_ylim([0,fs/2])
# plt.savefig(fig_name)

#trying to do 3d
fig_name="C:\\Users\\Harvey\\Desktop\\Uni\\avianz\\test plots"+"\\"+test_name+".jpg"
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(0, np.shape(coefs)[1])
Y = np.arange(0, np.shape(coefs)[0])
xx, yy = np.meshgrid(X, Y)
np.save("matrixharvey", coefs)
ax.plot_surface(xx, yy, coefs)
plt.show()
import WaveletSegment
import json
import wavio
import librosa
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.markers as mks
np.set_printoptions(suppress=True)
def testNodeEng(file):
   speciesData = {'Name': 'Kiwi', 'SampleRate': 16000, "TimeRange": [6, 32], "FreqRange": [800, 8000], "WaveletParams": [0.5, 0.5, [35, 36, 39, 40, 43, 44, 45]]}    # kiwi filter
   #speciesData = {'Name': 'Kiwi', 'SampleRate': 16000, "TimeRange": [6, 300], "FreqRange": [800, 8000], "WaveletParams": [0.5, 0.5, [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]]}
   with open(file + '.data') as f:
       print(file)
       segments = json.load(f)
       i = 0
       Eng = [None]* (len(segments)-1)
       for seg in segments:
           if seg[0] == -1:
               continue
           else:
               #  Read the segment
               wavobj = wavio.read(file, seg[1]-seg[0], seg[0])
               sampleRate = wavobj.rate
               data = wavobj.data
               if data.dtype != 'float':
                   data = data.astype('float')  # / 32768.0
               if np.shape(np.shape(data))[0] > 1:
                   data = np.squeeze(data[:, 0])
               if sampleRate != 16000:
                   data = librosa.core.audio.resample(data, sampleRate, 16000)
                   sampleRate = 16000
               # TODO: add denoid=sing and see change in res
               # Wavelet energy
               WS = WaveletSegment.WaveletSegment(speciesData)
               WS.sampleRate = sampleRate
               WS.data = data
               #print(len(data), sampleRate,seg[1]-seg[0])
               WE = WS.computeWaveletEnergy(data, sampleRate, 5, 'new', window=1, inc=1)
               #print(np.shape(WE))
               WE = np.mean(WE,axis=1)
               #print(np.shape(WE))
               #print(seg)
               #print("Energy in nodes 35, 36, 39, 40, 43, 44, 45 :", np.round(WE[34], 2), '\t', np.round(WE[35],2), '\t', np.round(WE[38],2), '\t', np.round(WE[39],2), '\t', np.round(WE[42],2), '\t', np.round(WE[43],2), '\t', np.round(WE[44],2))
               Eng[i] = [WE[node-1] for node in speciesData['WaveletParams'][2]]
               total = np.sum(Eng[i])
               Eng[i] /= total
               # .2, .2, .2, .2., .4, .5
               if ((Eng[i][0]>0.19) or (Eng[i][1]>0.19)) and ((Eng[i][4]>0.19) or (Eng[i][5]>0.19)) or (Eng[i][0]>0.25 or Eng[i][4]>0.25):
                   print(i," is a kiwi")
               else:
                   print(i," nah")
               i += 1
       #print(Eng)
       #print('max: ', np.round(np.max(Eng), 2))
       mx = np.max(Eng)
       md = np.median(Eng)

       x = [n+1 for n in range(i)]
       y = speciesData['WaveletParams'][2]
       pl.style.use('ggplot')
       valid_markers = ([item[0] for item in mks.MarkerStyle.markers.items() if
                         item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith(
                             'caret')])
       markers = np.random.choice(valid_markers, i, replace=False)
       pl.figure()
       # fig, ax = pl.subplots()
       for j in x:
           pl.plot(y, Eng[j-1], marker=markers[j-1], label='Seg' + str(x[j-1]))
       pl.plot(y, np.ones(np.shape(y))*mx, 'k--')
       pl.plot(y, np.ones(np.shape(y))*md, 'c--')
       pl.legend(x)
       pl.savefig(file+'b.png')
       #pl.show()


#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/kiwi.wav')
#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180413_184004.wav')
#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180504_185004.wav')
#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180517_215007.wav')
#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180502_192004.wav')
#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180503_190504.wav')
#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180501_182004.wav')
#testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180501_202004.wav')
testNodeEng('/Users/marslast/Projects/AviaNZ/Sound Files/20180502_185004.wav')

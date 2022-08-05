
# Given a folder that contains more folders birdID/year/... 
# each with wavs and data files, extract each syllable and save 
# each as birdID_year_callID_syllID

# This can then be used to extract the curve, and whatever else we want

import SignalProc 
import Segment
import os
import string, math
import wavio
import numpy as np
import pylab as pl
from ext import ce_denoise

def saveSyllables(path):
    ww = 256
    incr = 128
    sp = SignalProc.SignalProc()

    birds = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.isdigit()]
    print(birds)

    extracted = os.path.join(path,"extracted")
    try:
        os.mkdir(extracted)
    except: 
        print("Folder already exists")

    # For each folder
    for bird in birds:
        birddir = os.path.join(path,bird)
        # For each year
        years = [d for d in os.listdir(birddir) if os.path.isdir(os.path.join(birddir, d))]
        print(years)
        for year in years:
            # For each file
            yeardir = os.path.join(birddir,year)
            files = [f for f in os.listdir(yeardir) if f.lower().endswith(".wav")]
            print(files)
            callID = 0
            for call in files:
                # Load wav and data file
                filename = os.path.join(yeardir,call)
                sp.readWav(filename)
                # TODO: other params here
                sg = sp.spectrogram(ww,incr)
                #print(np.shape(sg))
                #pl.imshow(10*np.log10(np.flipud(sg.T)))
                
                segments = Segment.SegmentList()
                if os.path.isfile(filename + '.data'):
                    segments.parseJSON(filename+'.data', sp.fileLength / sp.sampleRate)
                else:
                    print("No data file ",filename)
    
                # Extract each syllable
                syllID = 0
                for s in segments:
                    print(s)
                    
                    # Extract sound and spectrogram
                    #print(s[0],s[1],math.floor(s[0]*incr),math.floor(s[1]*incr))
                    audio = sp.data[int(s[0]*sp.sampleRate):int(s[1]*sp.sampleRate)]
                    visual = sg[int(math.floor(s[0]*sp.sampleRate/incr)):int(math.floor(s[1]*sp.sampleRate/incr)),:]

                    # Fundamental frequency
                    #Wsamples = 4*incr
                    #thr = 0.5
                    #pitch = ce_denoise.FundFreqYin(audio, Wsamples, thr, sp.sampleRate)

                    # Choose name
                    name = os.path.join(extracted,bird+"_"+year+"_"+ format(callID,'03d')+"_"+ format(syllID,'03d'))
                    print(name)
    
                    # Save wav and image
                    wavio.write(name+".wav", audio.astype('int16'), int(sp.sampleRate), scale='dtype-limits', sampwidth=2)
                    np.save(name+".npy",visual)

                    fig = pl.imshow(10*np.log10(np.flipud(visual.T)))
                    pl.savefig(name+".png")

                    syllID += 1
                callID += 1
                
saveSyllables("\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID")
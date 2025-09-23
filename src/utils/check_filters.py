
import numpy as np

def convertHztoMel(f):
    return 1125*np.log(1+f/700)

def convertMeltoHz(m):
    return 700*(np.exp(m/1125)-1)

def mel_filter(filter='mel',nfilters=40,minfreq=0,maxfreq=None,normalise=False):
    # Transform the spectrogram to mel or bark scale
    if maxfreq is None:
        maxfreq = 8000
    print(filter,nfilters,minfreq,maxfreq,normalise)

    if filter=='mel':
        filter_points = np.linspace(0, 1125*np.log(1+maxfreq/700), nfilters + 2)  
        bins = convertMeltoHz(filter_points)

    nfft = 256
    freq_points = np.linspace(minfreq,maxfreq,nfft)

    filterbank = np.zeros((nfft,nfilters))
    for m in range(nfilters):
        # Find points in first and second halves of the triangle
        inds1 = np.where((freq_points>=bins[m]) & (freq_points<=bins[m+1]))
        inds2 = np.where((freq_points>=bins[m+1]) & (freq_points<=bins[m+2]))
        # Compute their contributions
        filterbank[inds1,m] = (freq_points[inds1] - bins[m]) / (bins[m+1] - bins[m])   
        filterbank[inds2,m] = (bins[m+2] - freq_points[inds2]) / (bins[m+2] - bins[m+1])             

    if normalise:
        # Normalise to unit area if desired
        norm = filterbank.sum(axis=0)
        norm = np.where(norm==0,1,norm)
        filterbank /= norm

    return filterbank


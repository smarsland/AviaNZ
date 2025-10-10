
# Time-frequency representations
# There are basically two ideas in here:
	# 1. We are interested in how a signal changes in time, particularly how the spectral density changes wrt time. 
		#So calculate the autocorrelation in terms of average time and time lag, and Fourier transform that. 
		#Turns out this is called the Wigner-Villes distribution. It would have been useful to know this first. 
		#It has cross-terms. 
		#These can be removed (or, at least, reduced) using windowing (=pseudo WVD) 
		#And it can be smoothed. 
		#It is an example of Cohen's class of bilinear time-freq representations, as is the spectrogram.
	# 2. These methods can be reassigned by moving the values of each bin from (t,nu) to the centre of gravity of the signal energy distribution around (t,nu). 
		#This improves the localisation.
		#Ridges can then be extracted.

# Basic problem is computational cost -- builds an N*N natrix 

# There is also a jupyter notebook
# There is still some debugging to do, but it is largely OK. And the results of test5 and test6 are promising

import numpy as np
import pylab as pl
from scipy.fftpack import fft, fftshift

def linSignal(N,i=0,f=0.5):
	# Create a linear signal
	t0 = round(N/2)
	y = np.ones((N,1))
	y[:,0] = np.arange(N)+1
	y = i*(y-t0) + (f-i)/(2*(N-1)) * ((y-1)**2 - (t0-1)**2)
	y = np.exp(2*np.pi*np.complex(0,1)*y)
	y = y/y[int(t0)-1]
	return y

def get_window(window_width=27,alpha=0.5):
    # Hamming (alpha=0.54) or Hanning (alpha=0.5) window
    beta = 1.0-alpha
    window_width += 1-np.fmod(window_width,2)
    h = alpha - beta*np.cos(2.0*np.pi*(np.arange(window_width)+1) / (window_width+1))
    Lh = len(h)//2
    h = h / h[Lh]
    return h, Lh

def diff_window(h):
    # Differentiate the window
    Lh = len(h)//2
    step = 0.5*(h[0]+h[-1])
    ramp = (h[-1]-h[0])//len(h)
    h2 = np.concatenate([np.zeros(1),h-step-ramp*np.arange(-Lh,Lh+1),np.zeros(1)])
    Dh = 0.5*(h2[2:len(h)+2] - h2[:len(h)]) + ramp
    Dh[0] += step
    Dh[-1] -= step
    return Dh

def spec(s,width=0):
    # Simple spectrogram
    r,c = np.shape(s)
    tfr = np.zeros((r,r),dtype=complex)

    if width==0:
        width=len(s)//4
    h, Lh = get_window(width)

    for i in range(r):
        taumin = -min(i,round(r/2)-1,Lh)
        taumax = min(r-i-1,round(r/2)-1,Lh)
        tau = np.arange(taumin,taumax+1)
        inds = np.fmod(tau+r,r)
        tfr[inds,i] = s[i+tau,0] * np.conj(h[Lh+tau])/(np.linalg.norm(h[Lh+tau]))
    tfr = np.abs(fft(tfr,axis=0))**2
    return tfr

def wvd(s):
    # Compute the Wigner-Villes distribution
    r,c = np.shape(s)
    tfr = np.zeros((r,r),dtype=complex)

    for i in range(r):
        taumax = min(i,r-i-1,round(r/2)-1)
        tau = np.arange(-taumax,taumax+1)
        inds = np.fmod(tau+r,r)
        tfr[inds,i] = s[i+tau,0] * np.conj(s[i-tau,c-1])
        tau = round((r-1)/2)
        if i <= r-tau & i>=tau+1:
            tfr[tau+1,i] = 0.5*(s[i+tau,0] * np.conj(s[i-tau,c]) + s[i-tau,1] * np.conj(s[i+tau,c-1]))
    tfr = fft(tfr)
    #pl.figure()
    #pl.imshow(tfr)
    return tfr

def pwvd(s):
    # Compute the pseudo-Wigner-Villes distribution (i.e., time-windowed version of the Wigner-Villes distribution)
    r,c = np.shape(s)
    tfr = np.zeros((r,r),dtype=complex)

    h, Lh = get_window(int(np.floor(len(s)/4)))

    for i in range(r):
        taumax = min(i,r-i-1,round(r/2)-1,Lh)
        tau = np.arange(-taumax,taumax+1)
        #inds = (r-1-tau)%(r-1)
        #inds = inds[::-1]
        inds = np.fmod(tau+r,r)
        #print(np.isreal(s[i+tau,0]*np.conj(s[i-tau,c-1])))
        tfr[inds,i] = h[Lh+tau] * s[i+tau,0] * np.conj(s[i-tau,c-1])
        tau = round((r-1)/2)
        if i <= r-tau & i>=tau+1 & tau<=Lh-1:
            tfr[tau+1,i] = 0.5*(h[Lh+tau] * s[i+tau,0] * np.conj(s[i-tau,c]) + h[Lh-tau] * s[i-tau,0] * np.conj(s[i+tau,c-1]))
    tfr = np.real(fft(tfr,axis=0))
    return tfr

def reassign_spec(s,width=0):
    # Spectrogram with reassignment
    r,c = np.shape(s)
    tfr1 = np.zeros((r,r),dtype=complex)
    tfr2 = np.zeros((r,r),dtype=complex)
    tfr3 = np.zeros((r,r),dtype=complex)

    if width==0:
        width = int(np.floor(len(s)/4))
        width = width+1-np.fmod(width,2)
    h, Lh = get_window(width,alpha=0.54)
    Th = h*np.arange(-Lh,Lh+1)
    Dh = diff_window(h)

    for i in range(r):
        taumin = -min(i,round(r/2)-1,Lh)
        taumax = min(r-i-1,(r/2)-1,Lh)
        tau = np.arange(taumin,taumax+1)
        inds = np.fmod(tau+r,r)
        tfr1[inds,i] = s[i+tau,0] * np.conj(h[Lh+tau])/np.linalg.norm(h[Lh+tau])
        tfr2[inds,i] = s[i+tau,0] * np.conj(Th[Lh+tau])/np.linalg.norm(h[Lh+tau])
        tfr3[inds,i] = s[i+tau,0] * np.conj(Dh[Lh+tau])/np.linalg.norm(h[Lh+tau])

    tfr1 = fft(tfr1,axis=0)
    tfr2 = fft(tfr2,axis=0)
    tfr3 = fft(tfr3,axis=0)

    tfr1.flatten()
    tfr2.flatten()
    tfr3.flatten()

    avoid_warn = np.where(tfr1!=0)
    tfr2[avoid_warn] = np.round(np.real(tfr2[avoid_warn]/tfr1[avoid_warn]))
    tfr3[avoid_warn] = np.round(np.imag(r*tfr3[avoid_warn]/tfr1[avoid_warn]/(2.0*np.pi)))
    tfr1 = np.abs(tfr1)**2

    tfr1 = np.reshape(tfr1,(r,r))
    tfr2 = np.reshape(tfr2,(r,r))
    tfr3 = np.reshape(tfr3,(r,r))
            
    rtfr = np.zeros((r,r),dtype=complex)
    Ex = np.mean(np.abs(s)**2)
    thr = (1.0e-6)*Ex
    for i in range(r):
        for j in range(r):
            if np.abs(tfr1[j,i])>thr:
                icol = i+tfr2[j,i]
                icol = int(min(max(icol,0),r-1))
                jcol = j - np.real(tfr3[j,i])
                jcol = int(np.fmod(np.fmod(jcol,r)+r,r))
                rtfr[jcol,icol] = rtfr[jcol,icol] + tfr1[j,i]
                tfr2[j,i] = jcol + np.complex(0,1)*icol
            else:
                tfr2[j,i] = np.inf*(1+np.complex(0,1))
                rtfr[j,i] = rtfr[j,i] + tfr1[j,i]
    return tfr1, rtfr #tfr2 #, tfr3, rtfr, tfr2

def rs(s,samplerate,width=0):
    # This is meant to be a faster version of spectrogram reassignment
    r = len(s)
    #r,c = np.shape(s)

    if width==0:
        width = int(np.floor(len(s)/4))
        width = width+1-np.fmod(width,2)
    #h, Lh = get_window(width,alpha=0.54)
    h = getweight()

    #for i in range(0,r-width-1,int(width/2)):
    for i in range(1):
        S = np.fft.fft(s[i:i+width]*h)
        Sdel = np.fft.fft(np.roll(s[i:i+width]*h,1))
        #Sdel = fft(np.roll(s[i:i+width]*window),1,axis=0)
        Sfreq = np.roll(S,1)
        C = S*np.conj(Sdel)
        C = C[:width//2]
        #freqs = np.angle(C)/(2*np.pi)
        freqs = samplerate*np.angle(C)/(2*np.pi)
        L = S*np.conj(Sfreq)
        L = L[:width//2]
        LGD =  - width/(2*np.pi*samplerate) * np.angle(L)
        #LGD = 0.5 - np.mod(1/(2*np.pi) * np.angle(L),1)
        #times = i + width/(2*samplerate) + LGD*width/samplerate
        times = i + LGD[i] + width/(2*samplerate)

        # Now need to resample onto the regular grid -> histogram2d -> range?
        spec, xedges, yedges = np.histogram2d(times,freqs,bins=(64,32),weights=S[:width//2])

    return S, C,L,LGD,times,freqs, spec
    

def reassign_pwvd(s,width=0):
    # Pseudo Wigner-Villes with reassignment

    r,c = np.shape(s)
    tfr1 = np.zeros((r,r),dtype=complex)
    tfr2 = np.zeros((r,r),dtype=complex)

    if width==0:
        width = int(np.floor(len(s)/4))
        width = width+1-np.fmod(width,2)
    h, Lh = get_window(width,alpha=0.54)
    Dh = diff_window(h)
        
    for i in range(r):
        taumax = min(i,r-i-1,round(r/2)-1,Lh)
        tau = np.arange(-taumax,taumax+1)
        inds = np.fmod(tau+r,r)
        tfr1[inds,i] = h[Lh+tau] * s[i+tau,0] * np.conj(s[i-tau,0])
        tfr2[inds,i] = Dh[Lh+tau] * s[i+tau,0] * np.conj(s[i-tau,0])
        tau = round((r-1)/2)
        if i <= r-tau & i>=tau & tau<=Lh-1:
            tfr1[tau,i] = 0.5*(h[Lh+tau] * s[i+tau,0] * np.conj(s[i-tau,c]) + h[Lh-tau] * s[i-tau,1] * np.conj(s[i+tau,c-1]))
            tfr2[tau,i] = 0.5*(Dh[Lh+tau] * s[i+tau,0] * np.conj(s[i-tau,c]) + Dh[Lh-tau] * s[i-tau,1] * np.conj(s[i+tau,c-1]))
    tfr1 = np.real(fft(tfr1,axis=0))
    tfr2 = np.imag(fft(tfr2,axis=0))

    tfr1.flatten()
    tfr2.flatten()

    avoid_warn = np.where(tfr1!=0)
    tfr2[avoid_warn] = np.round(r*np.real(tfr2[avoid_warn]/tfr1[avoid_warn]/(2.0*np.pi)))

    tfr1 = np.reshape(tfr1,(r,r))
    tfr2 = np.reshape(tfr2,(r,r))

    rtfr = np.zeros((r,r),dtype=complex)
    Ex = np.mean(np.abs(s)**2)
    thr = (1.0e-6)*Ex
    for i in range(r):
        for j in range(r):
            if np.abs(tfr1[j,i])>thr:
                jcol = j - tfr2[j,i]
                jcol = int(np.fmod(np.fmod(jcol,r)+r,r))
                rtfr[jcol,i] = rtfr[jcol,i] + tfr1[j,i]
                tfr2[j,i] = jcol
            else:
                tfr2[j,i] = np.inf
                rtfr[j,i] = rtfr[j,i] + tfr1[j,i]
    return rtfr, tfr2

def smooth_pwvd(s,width=0):
    # Smoothed pseudo Wigner-Villes

    r,c = np.shape(s)
    tfr = np.zeros((r,r),dtype=complex)

    if width==0:
        width = int(np.floor(len(s)/4.))
        width = width+1-np.fmod(width,2)
    h, Lh = get_window(width,alpha=0.54)

    # Smoothing window
    gwidth = int(np.floor(len(s)/10.))
    gwidth = gwidth+1-np.fmod(gwidth,2)
    g, Lg = get_window(gwidth,alpha=0.54)

    for i in range(r):
        taumax = min(i+Lg,r-i-1+Lg,round(r/2)-1,Lh)
        #print(i, taumax)
        points = np.arange(-min(Lg,r-i-1),min(Lg,i)+1)
        g2 = g[Lg+points]
        g2 = g2/np.sum(g2)
        tfr[0,i] = np.sum(g2*s[i-points,0] * np.conj(s[i-points,-1]))

        for tau in range(taumax):
                points = np.arange(-min(Lg,r-i-tau-2),min(Lg,i-tau-1)+1)
                g2 = g[Lg+points]
                g2 = g2/np.sum(g2)
                #print(i, tau, taumax, Lg, r-i-tau-2, points, g2)
                R1 = np.sum(g2*s[i+tau+1-points,0] * np.conj(s[i-tau-1-points,-1]))
                tfr[tau+1,i] = h[Lh+tau+1]*R1
                R2 = np.sum(g2*s[i-tau-1-points,0] * np.conj(s[i+tau+1-points,-1]))
                tfr[r-tau-1,i] = h[Lh-tau-1]*R2
                #print(points)
                #print(g2.T)
                #print(R1, R2, tfr[tau+1,i], tfr[r-tau-1,i], tau+1, r-tau-1, Lh+tau+1, Lh-tau-1)

        tau = round(r/2)
        if i <= r-tau & i>=tau+1 & tau<=Lh:
                points = np.arange(-min(Lg,r-i-tau-2),min(Lg,i-tau-1)+1)
                g2 = g[Lg+points]
                g2 = g2/np.sum(g2)
                tfr[tau+1,i] = 0.5*(g2*s[i+tau+1-points,0] * np.conj(s[i-tau-1-points,-1]) + g2*s[i-tau-1-points,0] * np.conj(s[i+tau+1-points,-1]))

    tfr = np.real(fft(tfr,axis=0))
    return tfr

def smooth_reassign_pwvd(s,width=0):
    #**** TO BE DEBUGGED
    # Smoothed, reassigned pseudo Wigner-Villes

    r,c = np.shape(s)
    tfr1 = np.zeros((r,r),dtype=complex)
    tfr2 = np.zeros((r,r),dtype=complex)
    tfr3 = np.zeros((r,r),dtype=complex)
    
    if width==0:
        width = int(np.floor(len(s)/4.))
        width = width+1-np.fmod(width,2)
    h, Lh = get_window(width,alpha=0.54)
    Dh = diff_window(h)

    # Smoothing window
    gwidth = int(np.floor(len(s)/10.))
    gwidth = gwidth+1-np.fmod(gwidth,2)
    g, Lg = get_window(gwidth,alpha=0.54)

    for i in range(r):
        taumax = min(i+Lg,r-i-1+Lg,round(r/2)-1,Lh)
        #print(i, taumax)
        points = np.arange(-min(Lg,r-i-1),min(Lg,i)+1)
        g2 = g[Lg+points]
        g2 = g2/np.sum(g2)
        Tg2 = g2 * points.T
        xx = s[i-points,0] * np.conj(s[i-points,-1])
         
        tfr1[0,i] = np.sum( g2 * xx) 
        tfr2[0,i] = np.sum( Tg2 * xx)
        tfr3[0,i] = Dh[Lh] * tfr1[0,i]

        for tau in range(taumax):
                points = np.arange(-min(Lg,r-i-tau-2),min(Lg,i-tau-1)+1)
                g2 = g[Lg+points]
                g2 = g2/np.sum(g2)
                Tg2 = g2 * points.T
                #print(i, tau, taumax, Lg, r-i-tau-2, points, g2)
                xx=s[i+tau+1-points,0] * np.conj(s[i-tau-1-points,-1])

                tfr1[tau+1,i] = np.sum(g2*xx)
                tfr2[tau+1,i] = h[Lh+tau+1]*np.sum(Tg2*xx)
                tfr3[tau+1,i] = Dh[Lh+tau+1]*tfr1[tau+1,i]
                tfr1[tau+1,i] = h[Lh+tau+1]*tfr1[tau+1,i]
                
                tfr1[r-tau-1,i] = np.sum(g2*np.conj(xx))
                tfr2[r-tau-1,i] = h[Lh-tau-1]*np.sum(Tg2*np.conj(xx))
                tfr3[r-tau-1,i] = Dh[Lh-tau-1]*tfr1[r-tau-1,i]
                tfr1[r-tau-1,i] = h[Lh-tau-1]*tfr1[r-tau-1,i]               

    tfr1 = fft(tfr1,axis=0)
    tfr2 = fft(tfr2,axis=0)
    tfr3 = fft(tfr3,axis=0)

    tfr1.flatten()
    tfr2.flatten()
    tfr3.flatten()

    avoid_warn = np.where(tfr1!=0)
    tfr2[avoid_warn] = np.round(np.real(tfr2[avoid_warn]/tfr1[avoid_warn]))
    tfr3[avoid_warn] = np.round(np.imag(r*tfr3[avoid_warn]/tfr1[avoid_warn]/(2.0*np.pi)))

    tfr1 = np.reshape(tfr1,(r,r))
    tfr2 = np.reshape(tfr2,(r,r))
    tfr3 = np.reshape(tfr3,(r,r))

    rtfr = np.zeros((r,r),dtype=complex)
    Ex = np.mean(np.abs(s)**2)
    thr = (1.0e-6)*Ex
    for i in range(r):
        for j in range(r):
            if np.abs(tfr1[j,i])>thr:
                icol = i-tfr2[j,i]
                icol = int(min(max(icol,0),r-1))
                jcol = j - np.real(tfr3[j,i])
                jcol = int(np.fmod(np.fmod(jcol,r)+r,r))
                rtfr[jcol,icol] = rtfr[jcol,icol] + tfr1[j,i]
                tfr2[j,i] = jcol + np.complex(0,1)*icol
            else:
                tfr2[j,i] = np.inf*(1+np.complex(0,1))
                rtfr[j,i] = rtfr[j,i] + tfr1[j,i]
    
    return rtfr, tfr2, tfr3

def ridges(tfr,tfrr,spec=False,smoothed=False):
    # Find ridges in the time-frequency distribution based on the stationary points of the reassignment operators

    r,c = np.shape(tfr)
    thr = np.sum(tfr)*0.5/(r*c)

    pointst = np.array([])
    pointsf = np.array([],dtype=float)

    for i in range(r):
        if spec | smoothed:
            # The smoothed version and the spectrogram have complex values for the reassignment
            inds = np.where((tfr[:,i]>thr) & ((np.real(tfrr[:,i])-np.arange(r))==0) & ((np.imag(tfrr[:,i])-i)==0))
        else:
            # The normal PWV has real values for the reassignment
            inds = np.where((tfr[:,i]>thr) & ((tfrr[:,i]-np.arange(r))==0))
        
        inds = inds[0]
        
        if len(inds)>0:
            pointst = np.append(pointst,np.ones((len(inds),1))*i)
            inds = inds.astype(float)
            if spec:
                inds = inds/float(r)
            else:
                inds = inds/(2.0*r)
            pointsf = np.append(pointsf,inds)

            #print(len(inds), len(pointst), len(pointsf), np.shape(pointst), np.shape(pointsf))
    pl.figure()
    pl.plot(pointst,pointsf,'.')
    pl.axis([0, r, 0, 0.5])
    pl.title('Ridges')
    return pointst, pointsf

def showspec(s,log=False,threshold=5,title='',smoothed=False): 
    pl.figure()
    mini=max(np.min(s),np.max(s)*threshold/100.0);
    inds = np.where(s<mini)
    s[inds] = mini
    if not smoothed:
        s = s[:np.shape(s)[0]//2,:]

    if log:
        s = 10.*np.log10(s)
    pl.imshow(np.flipud(s/np.max(s)))
    pl.title(title)

def test1():
    # Make a signal, plot it
    sig = linSignal(128)
    pl.ion()
    pl.figure()
    pl.plot(np.real(sig))

    # Plot the energy spectrum
    pl.figure()
    pl.plot(np.arange(-64,64)/128.0,np.squeeze(fftshift(np.abs(fft(sig.T))**2)))

    s = spec(sig)
    showspec(s,title='FFT')

    # Plot the PWVD
    s = pwvd(sig)
    showspec(s,title='PWVD')

def test2():
    import scipy.io as sio
    # Load the bat signal
    sig = sio.loadmat('/Users/marslast/Bibliographies/Code/tftb-0.2/data/bat.mat')
    sig = sig['bat']
    Fr = 230.4
    t0 = 2047

    pl.figure()
    pl.plot(np.real(sig))

    # Plot the energy spectrum
    pl.figure()
    s = np.squeeze(fftshift(np.abs(fft(sig.T))**2))
    s = s/max(s)
    pl.plot(s)

    s = spec(sig)
    showspec(s,title='FFT')
    # Plot the PWVD
    #from scipy.signal import hilbert
    #s = pwvd(hilbert(sig))
    #showspec(s,title='PWVD')   
    #pl.plot(np.arange(-64,64)/128.0,np.squeeze(fftshift(np.abs(fft(sig.T))**2)))

    # Plot the PWVD
    s = pwvd(sig)
    showspec(s,title='PWVD')

def test3():
    import scipy.io as sio
    # Load the gabor signal
    sig = sio.loadmat('/Users/marslast/Bibliographies/Code/tftb-0.2/data/gabor.mat')
    sig = sig['gabor']

    #pl.figure()
    #pl.plot(np.real(sig))

    # Plot the energy spectrum
    #pl.figure()
    #s = np.squeeze(fftshift(np.abs(fft(sig.T))**2))
    #s = s/max(s)
    #pl.plot(s)

    # Plot the spectrogram
    s = spec(sig,60)
    showspec(s,title='Spec')

    # Plot the PWVD
    from scipy.signal import hilbert
    s = pwvd(hilbert(sig))
    showspec(s,title='PWVD')

    s= pwvd(sig)
    showspec(s,title="PWVD")

def test4():
    # Test the reassignment
    N=128
    t = np.ones((N,1))
    t[:,0] = np.arange(N)
    phi = np.arccos((1.0/1.5))
    phase = 2.0*np.pi*0.3*t + 0.5*0.3*100*(np.sin(2.0*np.pi*t/100.0+phi) - np.sin(phi))
    sig1 = np.exp(np.complex(0,1)*phase)

    c = (0.05-0.5)/(1./32 - 1./1)
    f0 = 0.5-c
    t[:,0] = np.arange(N)+1
    phi = 2.0*np.pi*(f0*t + c*np.log(np.abs(t)))
    sig2 = np.exp(np.complex(0,1)*phi)

    sig = sig1+sig2

    s = spec(sig,width=27)
    showspec(np.real(s),title='Spec')

    s,t = reassign_spec(sig)
    showspec(np.abs(s),title='Reassigned Spec')

def test5():
    # Test the reassignment 
    N=60
    t = np.ones((N,1))
    t[:,0] = np.arange(N)
    phase = 2.0*np.pi*0.25*t + 0.5*0.2*50*np.sin(2.0*np.pi*t/50.0)
    sig1 = np.exp(np.complex(0,1)*phase)

    sig2 = linSignal(N,0.3,0.1)

    t = t+1-int(N/2)
    sig3 = np.exp(np.complex(0,1)*2.0*np.pi*0.4*t)
    sig3 = sig3/sig3[int(N/2)-1]

    sig = np.concatenate([sig1, np.zeros((8,1)), sig2+sig3])

    s = spec(sig)
    showspec(np.abs(s),title='Spec')

    s,t = reassign_spec(sig)
    showspec(np.abs(s),title='Reassigned Spec')
    x,y=ridges(s,t,spec=True)

    s = wvd(sig)
    showspec(np.abs(s),title='PWVD')
    
    s = smooth_pwvd(sig)
    showspec(s,title='Smoothed PWVD',smoothed=True)

    pl.figure()
    pl.imshow(np.flipud(s))
    pl.title('test')
    
    s,t = reassign_pwvd(sig)
    s = np.abs(s)
    showspec(s,title='Reassigned PWVD',smoothed=True)
    a,b = ridges(s,t)
    
    s,t,u = smooth_reassign_pwvd(sig)
    s = np.abs(s)
    showspec(s,title='Smoothed Reassigned PWVD',smoothed=True)
    a,b = ridges(s,t,smoothed=True)
    

def test6():
    # Test the smoothing
    N = 128
    t = np.ones((N,1))
    t[:,0] = np.arange(N)-N/2+1   
    sig1 = np.exp(np.complex(0,1)*2.0*np.pi*0.15*t)
    sig1 = sig1/(sig1[N//2])
    
    sig3 = np.exp(np.complex(0,1)*2.0*np.pi*0.4*t)
    sig3 = sig3/sig3[N//2]

    sig2 = np.exp(-(t/(2.0*np.sqrt(N)))**2 * np.pi)
    sig2 = sig2*sig3

    E1 = np.mean(np.abs(sig1)**2)
    E2 = np.mean(np.abs(sig2)**2)
    h = np.sqrt(E1 / (E2*np.power(10,(5./10)) ))
    sig = sig1+h*sig2

    s = wvd(sig)
    showspec(np.abs(s),title='WVD')

    s = pwvd(sig)
    showspec(np.abs(s),title='PWVD')

    s = smooth_pwvd(sig)
    showspec(s,title='Smoothed PWVD',smoothed=True)
    
    s,t = reassign_pwvd(sig)
    s = np.abs(s)
    showspec(s,title='Reassigned PWVD',smoothed=True)
    a,b = ridges(s,t)
    
    s,t,u = smooth_reassign_pwvd(sig)
    s = np.abs(s)
    showspec(s,title='Smoothed Reassigned PWVD',smoothed=True)
    a,b = ridges(s,t,smoothed=True)
    
def test7():
    import soundfile as sf
    sig, sampleRate = sf.read('Sound Files/tril1.wav') 

    # None of the following should be necessary for librosa
    if sig.dtype is not 'float':
        sig = sig.astype('float') #/ 32768.0
    if np.shape(np.shape(sig))[0]>1:
        sig = sig[:,0]
    sig = sig.reshape(len(sig),1)

    s = wvd(sig)
    showspec(np.abs(s),title='WVD')

    s = pwvd(sig)
    showspec(np.abs(s),title='PWVD')

    s = smooth_pwvd(sig)
    showspec(s,title='Smoothed PWVD',smoothed=True)
    
    s,t = reassign_pwvd(sig)
    s = np.abs(s)
    showspec(s,title='Reassigned PWVD',smoothed=True)
    a,b = ridges(s,t)
    
    s,t,u = smooth_reassign_pwvd(sig)
    s = np.abs(s)
    showspec(s,title='Smoothed Reassigned PWVD',smoothed=True)

    a,b = ridges(s,t,smoothed=True)
    
def getweight():
    #return np.array([0.        , 0.015932  , 0.06311576, 0.13973802, 0.24285423, 0.3685017 , 0.51185186, 0.66739585, 0.8291562 , 0.99091655, 1.14646054, 1.2898107 , 1.41545817, 1.51857438, 1.59519664, 1.64238039, 1.6583124 , 1.64238039, 1.59519664, 1.51857438, 1.41545817, 1.2898107 , 1.14646054, 0.99091655, 0.8291562 , 0.66739585, 0.51185186, 0.3685017 , 0.24285423, 0.13973802, 0.06311576, 0.015932  , 0.        ]) 
    return np.array([ 0.00000000e+00, 1.00297754e-03, 4.00945571e-03, 9.01207715e-03, 1.59985996e-02, 2.49519259e-02, 3.58501457e-02, 4.86665893e-02, 6.33698927e-02, 7.99240744e-02, 9.82886236e-02, 1.18418599e-01, 1.40264740e-01, 1.63773584e-01, 1.88887602e-01, 2.15545335e-01, 2.43681548e-01, 2.73227386e-01, 3.04110547e-01, 3.36255452e-01, 3.69583440e-01, 4.04012950e-01, 4.39459728e-01, 4.75837029e-01, 5.13055832e-01, 5.51025057e-01, 5.89651786e-01, 6.28841494e-01, 6.68498276e-01, 7.08525086e-01, 7.48823971e-01, 7.89296314e-01, 8.29843072e-01, 8.70365019e-01, 9.10762993e-01, 9.50938132e-01, 9.90792121e-01, 1.03022743e+00, 1.06914756e+00, 1.10745725e+00, 1.14506277e+00, 1.18187209e+00, 1.21779512e+00, 1.25274395e+00, 1.28663307e+00, 1.31937953e+00, 1.35090321e+00, 1.38112696e+00, 1.40997681e+00, 1.43738217e+00, 1.46327597e+00, 1.48759484e+00, 1.51027927e+00, 1.53127375e+00, 1.55052690e+00, 1.56799161e+00, 1.58362514e+00, 1.59738922e+00, 1.60925018e+00, 1.61917899e+00, 1.62715136e+00, 1.63314776e+00, 1.63715354e+00, 1.63915888e+00, 1.63915888e+00, 1.63715354e+00, 1.63314776e+00, 1.62715136e+00, 1.61917899e+00, 1.60925018e+00, 1.59738922e+00, 1.58362514e+00, 1.56799161e+00, 1.55052690e+00, 1.53127375e+00, 1.51027927e+00, 1.48759484e+00, 1.46327597e+00, 1.43738217e+00, 1.40997681e+00, 1.38112696e+00, 1.35090321e+00, 1.31937953e+00, 1.28663307e+00, 1.25274395e+00, 1.21779512e+00, 1.18187209e+00, 1.14506277e+00, 1.10745725e+00, 1.06914756e+00, 1.03022743e+00, 9.90792121e-01, 9.50938132e-01, 9.10762993e-01, 8.70365019e-01, 8.29843072e-01, 7.89296314e-01, 7.48823971e-01, 7.08525086e-01, 6.68498276e-01, 6.28841494e-01, 5.89651786e-01, 5.51025057e-01, 5.13055832e-01, 4.75837029e-01, 4.39459728e-01, 4.04012950e-01, 3.69583440e-01, 3.36255452e-01, 3.04110547e-01, 2.73227386e-01, 2.43681548e-01, 2.15545335e-01, 1.88887602e-01, 1.63773584e-01, 1.40264740e-01, 1.18418599e-01, 9.82886236e-02, 7.99240744e-02, 6.33698927e-02, 4.86665893e-02, 3.58501457e-02, 2.49519259e-02, 1.59985996e-02, 9.01207715e-03, 4.00945571e-03, 1.00297754e-03, 0.00000000e+00])

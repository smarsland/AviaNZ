
from ext import ce_WignerVille as tf
import pylab as pl
import numpy as np
from scipy.fftpack import fft, fftshift

def test1():
    # Make a signal, plot it
    sig = tf.linSignal(128)
    pl.ion()
    pl.figure()
    pl.plot(np.real(sig))

    # Plot the energy spectrum
    pl.figure()
    pl.plot(np.arange(-64,64)/128.0,np.squeeze(fftshift(np.abs(fft(sig.T))**2)))

    # Plot the PWVD
    s = tf.pwvd(sig)
    tf.showspec(s,title='PWVD')

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

    # Plot the PWVD
    from scipy.signal import hilbert
    s = tf.pwvd(hilbert(sig))
    tf.showspec(s,title='PWVD')   
    #pl.plot(np.arange(-64,64)/128.0,np.squeeze(fftshift(np.abs(fft(sig.T))**2)))

    # Plot the PWVD
    s = tf.pwvd(sig)
    tf.showspec(s,title='PWVD')

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
    s = tf.spec(sig,60)
    tf.showspec(s,title='Spec')

    # Plot the PWVD
    from scipy.signal import hilbert
    s = tf.pwvd(hilbert(sig))
    tf.showspec(s,title='PWVD')

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

    s = tf.spec(sig,width=27)
    tf.showspec(np.real(s),title='Spec')

    s,t = tf.reassign_spec(sig)
    tf.showspec(np.abs(s),title='Reassigned Spec')

def test5():
    # Test the reassignment 
    N=60
    t = np.ones((N,1))
    t[:,0] = np.arange(N)
    phase = 2.0*np.pi*0.25*t + 0.5*0.2*50*np.sin(2.0*np.pi*t/50.0)
    sig1 = np.exp(np.complex(0,1)*phase)

    sig2 = tf.linSignal(N,0.3,0.1)

    t = t+1-int(N/2)
    sig3 = np.exp(np.complex(0,1)*2.0*np.pi*0.4*t)
    sig3 = sig3/sig3[int(N/2)-1]

    sig = np.concatenate([sig1, np.zeros((8,1)), sig2+sig3])

    s = tf.spec(sig)
    tf.showspec(np.abs(s),title='Spec')

    s,t = tf.reassign_spec(sig)
    tf.showspec(np.abs(s),title='Reassigned Spec')
    x,y=tf.ridges(s,t,spec=True)

    s = tf.wvd(sig)
    tf.showspec(np.abs(s),title='PWVD')
    
    s = tf.smooth_pwvd(sig)
    tf.showspec(s,title='Smoothed PWVD',smoothed=True)

    pl.figure()
    pl.imshow(np.flipud(s))
    pl.title('test')
    
    s,t = tf.reassign_pwvd(sig)
    s = np.abs(s)
    tf.showspec(s,title='Reassigned PWVD',smoothed=True)
    a,b = tf.ridges(s,t)
    
    s,t,u = tf.smooth_reassign_pwvd(sig)
    s = np.abs(s)
    tf.showspec(s,title='Smoothed Reassigned PWVD',smoothed=True)
    a,b = tf.ridges(s,t,smoothed=True)
    

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

    s = tf.wvd(sig)
    tf.showspec(np.abs(s),title='WVD')

    s = tf.pwvd(sig)
    tf.showspec(np.abs(s),title='PWVD')

    s = tf.smooth_pwvd(sig)
    tf.showspec(s,title='Smoothed PWVD',smoothed=True)
    
    s,t = tf.reassign_pwvd(sig)
    s = np.abs(s)
    tf.showspec(s,title='Reassigned PWVD',smoothed=True)
    a,b = tf.ridges(s,t)
    
    s,t,u = tf.smooth_reassign_pwvd(sig)
    s = np.abs(s)
    tf.showspec(s,title='Smoothed Reassigned PWVD',smoothed=True)
    a,b = tf.ridges(s,t,smoothed=True)
    
def test7():
    import wavio
    wavobj = wavio.read('Sound Files/tril1.wav')
    sampleRate = wavobj.rate
    sig = wavobj.data

    # None of the following should be necessary for librosa
    if sig.dtype is not 'float':
        sig = sig.astype('float') #/ 32768.0
    if np.shape(np.shape(sig))[0]>1:
        sig = sig[:,0]
    sig = sig.reshape(len(sig),1)

    s = tf.wvd(sig)
    tf.showspec(np.abs(s),title='WVD')

    s = tf.pwvd(sig)
    tf.showspec(np.abs(s),title='PWVD')

    s = tf.smooth_pwvd(sig)
    tf.showspec(s,title='Smoothed PWVD',smoothed=True)
    
    s,t = tf.reassign_pwvd(sig)
    s = np.abs(s)
    tf.showspec(s,title='Reassigned PWVD',smoothed=True)
    a,b = tf.ridges(s,t)
    
    s,t,u = tf.smooth_reassign_pwvd(sig)
    s = np.abs(s)
    tf.showspec(s,title='Smoothed Reassigned PWVD',smoothed=True)
    a,b = tf.ridges(s,t,smoothed=True)
    

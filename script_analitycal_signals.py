# 19/11/2021
# Author: Virginia Listanti

# In this script you can find code to generate and save analytical signals
# comment the parts you don't need
# LEGEND:
# Omega_0 is always the starting frequency of a chirp
# omega_1 is always the end frequency of a chirp
# phi_t is the phase law you need to generate the waveform of the signal
# inst_freq_func is the Instantaneous frequency law. Note: this is a lambda function

#NOTE:
#If you change the sample rate (fs) you can have sound only between [0,fs/2], so you have to change omega_0, omega_1
#accordigly

import numpy as np
import wavio
#from scipy.io import loadmat, savemat

T=5 #duration in seconds
A=1 #amplitude
phi=0 #initial phase. Usually 0
samplerate = 8000 #this is the sample rate in HZ(of course)
t = np.linspace(0., T, samplerate*T,endpoint=False)
file_name=' ' #insert path where you want to change the file


#pure tone

omega=1000 #tone frequency
phi_t= phi + 2. * np.pi * omega * t
inst_freq_fun= lambda t: 100*np.ones((np.shape(t)))

# #linear up-chirp
omega_1=2000
omega_0=500
c=(omega_1-omega_0)/T
phi_t=phi+2*np.pi*(0.5*c*t**2+omega_0*t)
inst_freq_fun=lambda x: omega_0+c*x

#linear down-chirp
omega_1=500
omega_0=2000
c=(omega_1-omega_0)/T
phi_t=2*np.pi*(0.5*c*t**2+omega_0*t)
inst_freq_fun=lambda x: omega_0+c*x

# #exponential up-chirp

omega_1=2000
omega_0=500
k=(omega_1/omega_0)**(1/T)
phi_t=phi+2*np.pi*omega_0*((k**t-1)/np.log(k))
inst_freq_fun=lambda x: omega_0*k**x

# #exponential down-chirp
omega_1=500
omega_0=2000
k=(omega_1/omega_0)**(1/T)
phi_t=phi+2*np.pi*omega_0*((k**t-1)/np.log(k))
inst_freq_fun=lambda x: omega_0*k**x

s1=A*np.sin(phi+phi_t) #this is the waveform of the signal
wavio.write(file_name,s1, samplerate,sampwidth=2)


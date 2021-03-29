
#23/03/2021
#Author: Virginia Listanti

# Extraxct ridge curve from sound file
# This code is taken from Iatsenko et al.

# -------------------------------Copyright----------------------------------
# Author: Dmytro Iatsenko
# Information about these codes (e.g. links to the Video Instructions),
# as well as other MatLab programs and many more can be found at
#  http://www.physics.lancs.ac.uk/research/nbmphysics/diats/tfr

# Related articles:
#  [1] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
# "Linear and synchrosqueezed time-frequency representations revisited.
#   Part I: Overview, standards of use, related issues and algorithms."
#  {preprint - arXiv:1310.7215}
# [2] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
#  "Linear and synchrosqueezed time-frequency representations revisited.
#  Part II: Resolution, reconstruction and concentration."
#  {preprint - arXiv:1310.7274}
# [3] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
#  "On the extraction of instantaneous frequencies from ridges in
#  time-frequency representations of signals."
#  {preprint - arXiv:1310.7276}
#


import numpy as np

class ecinfo:
    #reproduce homonimous structure in the original code
    def __init__(self,info1=[], info2=[]):
        self.info1=info1
        self.info2=info2


class IF:
    def __init__(self,method=2,NormMode='off',DispMode='on', PlotMode='off', PathOpt='on'):
        self.method=method
        if self.method==1:
            self.pars=1
        elif self.method==2:
            self.pars=[1,1]
        else:
            self.pars=[]
        self.NormMode=NormMode
        self.DispMode=DispMode
        self.PlotMode=PlotMode
        self.Skel=[]
        self.PathOpt=PathOpt
        #AmpFunc=@(x)log(x
        PenalFunc={[],[]}
        MaxIter=20



    def ecurve(TFR,freq,wopt,PropertyName,PropertyValue):
   
       # extracts the curve (i.e. the sequence of the amplitude ridge points)
       # and its full support (the widest region of unimodal TFR amplitude
       # around them) from the given time-frequency representation [TFR] of the
       # signal. TFR can be either WFT or WT.

       #OUTPUT:
       # tfsupp: 3xL matrix
       #    - extracted time-frequency support of the component, containing:
       #        frequencies of the TFR amplitude peaks (ridge points) in the first row (referred as \omega_p(t)/2\pi in [3])
       #        support lower bounds (referred as \omega_-(t)/2/pi in [1]) - in the second row,
       #        the upper bounds (referred as \omega_+(t)/2/pi in [1]) - in the third row.
       # ecinfo: structurecontains all the relevant information about the process of curve extraction.
       #         Is it a class?
       # Skel: 4x1 cell (returns empty matrix [] if 'Method' property is not 1,2,3 or 'nearest')
       #      - contains the number of peaks N_p(t) in [Skel{1}],
       #       their frequency indices m(t) in [Skel{2}],
       #        the corresponding frequencies \nu_m(t)/2\pi in [Skel{3}],
       #         and the respective amplitudes Q_m(t) in [Skel{4}] (in notations of [3]).
   

        # INPUT:
        # TFR: NFxL matrix (rows correspond to frequencies, columns - to time)
        #    - WFT or WT from which to extract [tfsupp]
        # freq: NFx1 vector
        #    - the frequencies corresponding to the rows of [TFR]
        # wopt: structure | value (except if 'Method' property is 1 or 3) | 1x2 vector
        #    - structure with parameters of the window/wavelet and the simulation, returned as a third output by functions wft.m and wt.m;
        #%      alternatively, one can set wopt=[fs], where [fs] is the signal
        #%      sampling frequency (except methods 1 and 3); for methods 1 and 3
        #%      one can set wopt=[fs,D], where [D] is particular parameter of the
        #%      method (see [3]): (method 1) the characteristic growth rate of the
        #%      frequency - df/dt (in Hz/s) - for the WFT, or of the log-frequency
        #%      - d\log(f)/dt - for the WT; (method 2) the minimal distinguishable
        #%      frequency difference (in Hz) for WFT, or log-difference for WT.
 


        [NF,L]=np.shape(TFR)

        #inizialization

        freq=np.reshape(freq,(1,NL)) #reshape freq as a column vector.

        #this vectors were inizialized multiplying by NaN. I don't think it is necessary
        tfsupp=np.zeros((3,L))
        pind=np.zeros((1,L)) 
        pamp=np.zeros((1,L))
        idr=np.zeros((1,L))
        dfreq=np.zeros((NF-1, 1))

        #check how to do this in Python
        #if nargout>1
        #ec=struct;
        #ec.Method=method; ec.Param=pars; ec.Display=DispMode; ec.Plot=PlotMode;
        #ec.Skel=Skel; ec.PathOpt=PathOpt; ec.AmpFunc=AmpFunc;
        #end


         #Determine the frequency resolution
         #MATLAB diff difference between adjacent elements along first array dim
         
        if np.min(freq)<=0 or np.std(np.diff(freq,1,0))<np.std(np.diff(log(freq1,1,0))):
            fres=1
            fstep=np.mean(np.diff(freq,1,0))
            dfreq[:,0]=freq[0]-freq[-1:1] 
            dfreq[:,1]=freq[-1]-freq[-1:1]
        else:
            fres=2
            fstep=np.mean(np.diff(log(freq,1,0)))
            dfreq[:,0]=log(freq[0])-log(freq[-1:1])
            dfreq[:,1]=log(freq[-1])-log(freq[-1:0])

        #Assign numerical parameters
        if type(wopt) is dict:
            fs=wopt.fs
            DT=(wopt.wp.t2h-wopt.wp.t1h)

            if fres==1:
                DF=(wopt.wp.xi2h-wopt.wp.xi1h)/2/pi
            else: 
                DF=log(wopt.wp.xi2h/wopt.wp.xi1h)

            if method==1:
               DD=DF/DT
            elif method==3: 
                DD=DF

        else:
            fs=wopt(1)
            if method==1 or method==3:
               DD=wopt(2)

        #//////////////////////////////////////////////////////////////////////////
        # TFR=abs(TFR) not needed. We already take absolute value in signalProc
        #convert to absolute values, since we need only them; also improves computational speed as TFR is no more complex and is positive

        nfunc=np.ones((NF,1))
        tn1=find(~isnan(TFR(end,:)),1,'first'); tn2=find(~isnan(TFR(end,:)),1,'last'); sflag=0;
    
    
        return tfsupp,ecinfo,Skel





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


################ Notes

#ind2sub => numpy.unravel_index
#sub2ind => numpy.ravel_multi_index

#######################################

#######################TODO: 
# X 1) check definition of class ok
# 2)check use of classes 
# X 3)check PenalFunction => PenalFunction is a dictionary with elements '1' and '2' done
# X 4) Review use of matlab function_handle => we need something more efficient
# X 5) check indexing
# 6) Check all find -> ravel_multi_index(np.nonzero
# 7) argmax -> np.ravel_index(np.argmax


import numpy as np
import math
import types
import scipy.fftpack as fft

#class ecinfo:
#    #reproduce homonimous structure in the original code
#    def __init__(self,info1=[], info2=[]):
#        self.info1=info1
#        self.info2=info2

class ec_class:
    #reproduce homonimous structure in the original code
    def __init__(self,efreq=[], eind=[], pfreq=[],pind=[], pamp=[], idr=[], mv=[], rdiff=[]):
        self.efreq=efreq 
        self.eind=eind
        self.pfreq=pfreq
        self.pind=pind
        self.pamp=pamp
        self.idr=idr
        self.mv=mv
        ec.rdiff=rdiff

class Wp:
    def __init__(self, window_length, SampleRate, window_name='Hann', f0=1):
        #%wp - structure with wavelet parameters containing fields:
        #%     xi1,xi2 - wavelet full support in the frequency domain;
        #%     ompeak,tpeak - wavelet peak frequency and peak time
        #%     t1,t2 - wavelet full support in the time domain;
        #%     fwtmax,twfmax - maximum value of abs(fwt) and abs(twf);
        #%     C,D - coefficients needed for reconstruction
        #[fwt] and [twf] - window function in frequency and time

        #logw1= lambda x, pars,fs, DD: (x)-pars(1)*abs(fs*x/DD)
        Wp.fwtmax=[]; Wp.twfmax=[]; Wp.C=[]; Wp.omg=[];
        Wp.xi1=-np.Inf; Wp.xi2=np.Inf; Wp.ompeak=[];
        Wp.t1=-np.Inf; Wp.t2=np.Inf; Wp.tpeak=[];
        if window_name=='Hann':
            q=4.4*f0;
            Wp.twf=lambda t: (1+np.cos(2*np.pi*t/q))/2
            Wp.t1=-q/2
            Wp.t2=q/2;
            Wp.fwt= lambda xi:(-(2*np.pi/q)**2)*np.sin(xi*q/2)/(xi*(xi**2-(2*np.pi/q)**2))
            Wp.ompeak=0
            Wp.C=np.pi*Wp.twf(0)
            Wp.omg=0
            Wp.tpeak=0;

        Wp.xi1=np.amin(Wp.fwt(np.arange(0,window_length*SampleRate,SampleRate)))
        Wp.xi2=np.amax(Wp.fwt(np.arange(0,window_length*SampleRate,SampleRate)))


        
        Wp.fwtmax=Wp.fwt(Wp.ompeak)
        if np.isnan(Wp.fwtmax):
            Wp.fwtmax=Wp.fwt(Wp.ompeak+10**(-14))

        Wp.twfmax=Wp.twf(np.arange(window_length))

        #DT=(wopt.wp.t2h-wopt.wp.t1h)

        #if fres==1:
        #    DF=(wopt.wp.xi2h-wopt.wp.xi1h)/2/pi
        #else: 
        #    DF=log(wopt.wp.xi2h/wopt.wp.xi1h)


class Wopt:
    #reproduce homonimous structure in the original code
    def __init__(self,sampleRate,fmin,fmax,wp,TFRname='FT',window_type='Hann'):
        Wopt.TFRname=TFRname
        #if TFRname=='WT':
        #    wopt.wp=wp; #parameters of the wavelet
            
        #    #wopt.PadLR={padleft,padright}; I don't thinkwe need it 
        #else:
        #    wp.fwt = func2str(wp.fwt)
        #    #varargout{2} = wp;

        Wopt.fs=sampleRate
        Wopt.window=window_type
        #wopt.f0=f0;
        Wopt.fmin=fmin;
        Wopt.fmax=fmax;
        Wopt.wp=wp
        #wopt.nv=nv; wopt.nvsim=nvsim;
        #wopt.Padding=PadMode;
        #wopt.RelTol=RelTol;
        #wopt.Preprocess=Preprocess;
        #wopt.Plot=PlotMode;
        #wopt.Display=DispMode;
        #wopt.CutEdges=CutEdges;




class IF:
    def __init__(self,method=2,NormMode='off',DispMode='on', PathOpt='on'):
        self.method=method
        if self.method==1:
            self.pars=1
        elif self.method==2:
            self.pars=[1,1]
        else:
            self.pars=[]
        self.NormMode=NormMode
        self.DispMode=DispMode
        #self.PlotMode=PlotMode not needed
        self.Skel={'Np':[],'mt':[],'nu':[],'qn':[]}
        self.PathOpt=PathOpt
        #AmpFunc=@(x)log(x
        self.PenalFunc={'1':[],'2':[]}
        self.MaxIter=20


    def AmpFunc(x):
        return x*log(x)


    def ecurve(self,TFR,freq,wopt):
   
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
       #      - contains the number of peaks N_p(t) in [Skel{1}], -> Skel['Np']
       #       their frequency indices m(t) in [Skel{2}], -> Skel['mt']
       #        the corresponding frequencies \nu_m(t)/2\pi in [Skel{3}], -> Skel['nu']
       #         and the respective amplitudes Q_m(t) in [Skel{4}] (in notations of [3]). -> Skel['qm']
   

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
        NaN=math.nan

        #inizialization

        freq=np.reshape(freq,(NF,1)) #reshape freq as a column vector.

        #this vectors were inizialized multiplying by NaN. I don't think it is necessary
        tfsupp=np.zeros((3,L))*NaN
        pind=np.zeros((1,L))*NaN 
        pamp=np.zeros((1,L))*NaN 
        idr=np.zeros((1,L))*NaN 
        dfreq=np.zeros((NF-1,2))

        #check how to do this in Python
        #if nargout>1
        #ec=struct;
        #ec.Method=method; ec.Param=pars; ec.Display=DispMode; ec.Plot=PlotMode;
        #ec.Skel=Skel; ec.PathOpt=PathOpt; ec.AmpFunc=AmpFunc;
        #end

        ecinfo={'Method':self.method, 'Param':self.pars, 'Display':self.DispMode, 'Skel':self.Skel, 'PathOpt':self.PathOpt}


            #Determine the frequency resolution
            #MATLAB diff difference between adjacent elements along first array dim
         
        if np.amin(freq)<=0 or np.std(np.diff(freq,1,0))<np.std(np.diff(np.log(freq1,1,0))):
            fres=1
            fstep=np.mean(np.diff(freq,1,0))
            print(np.shape(freq),np.shape(dfreq))
            dfreq[:,0]=freq[0,0]-freq[-1:0:-1,0] #different shape in original
            dfreq[:,1]=freq[-1,0]-freq[-2::-1,0]
        else:
            fres=2
            fstep=np.mean(np.diff(log(freq,1,0)))
            dfreq[:,0]=log(freq[0])-log(freq[-1:0:-1])
            dfreq[:,1]=log(freq[-1])-log(freq[-1::-1])

        #Assign numerical parameters
        if isinstance(wopt, Wopt):
            fs=wopt.fs
            DT=(wopt.wp.t2-wopt.wp.t1)

            if fres==1:
                DF=(wopt.wp.xi2-wopt.wp.xi1)/2/np.pi
            else: 
                DF=log(wopt.wp.xi2/wopt.wp.xi1)

            if self.method==1:
                DD=DF/DT
            elif self.method==3: 
                DD=DF

        else:
            fs=wopt[0]
            if self.method==1 or self.method==3:
                DD=wopt[1]

        #//////////////////////////////////////////////////////////////////////////
        # TFR=abs(TFR) not needed. We already take absolute value in signalProc
        #convert to absolute values, since we need only them; also improves computational speed as TFR is no more complex and is positive

        nfunc=np.ones((NF,1))
        if np.isnan(TFR[-1,:]).any():
            tn1=np.argwhere(np.isnan(TFR[-1,:]))[0] #find first index where TFR is NAN of last line:
        else:
            tn1=0
        if np.isnan(TFR[-1,:]).any():
            tn2=np.argwhere(np.isnan(TFR[-1,:]))[-1] #find last index where TFR is NAN of last line
        else:
            tn2=L
        sflag=0;


        

        if (type(self.method) is 'char' and self.method.lower!='max') or self.method==1: #if not frequency-based or maximum-based extraction
            sflag=1
            #----------------------------------------------------------------------
            #Construct matrices of ridge indices, frequencies and amplitudes:
            #[Ip],[Fp],[Qp], respectively; [Np] - number of peaks at each time.
            if self.Skel:
                Np=np.zeros((1,L))
                Ip=np.zeros((1,L))
                Fp=np.zeros((1,L))
                Qp=np.zeros((1,L))
                Mp=np.amax(Np)
            else:
                if self.DispMode.lower!='off' and self.DispMode.lower!='notify':
                    print('Locating the amplitude peaks in TFR... ')
                TFR=np.stack((zeros(1,L),TFR,zeros(1,L))) #pad TFR with zeros

                #find indices of the peaks 
                idft=1+np.argwhere(TFR.flatten()[1:-1]>=TFR.flatten()[0:-2] and TFR.flatten()[1:-1]>TFR.flatten()[2:]) #flatten TFR
                #separate frequency and time indices of the peaks
                [idf,idt]=np.unravel_index(idft,np.shape(A))
                #idf=idft[:,0]
                #idt=idft[:,1]
                #idf=idf-1 #do we need it?
                idb=np.argwhere(idf==0 or idf==NF-1)
                #remove the border peaks
                idft[idb]=[] 
                idf[idb]=[]
                idt[idb]=[]; 
                dind=[0, np.argwhere(np.diff(idt,1,0)>0),len(idt)]
                Mp=np.amax([np.amax(np.diff(dind,1,0)),2])
                Np=np.zeros((1,L))
                idn=np.zeros((len(idt),1))
                for dn in range(len(dind)):
                    ii=np.arange(dind[dn]+1,dind[dn+1]+1,1)
                    idn[ii]=np.arange(0,len[ii]+1,1) #changef 1 in 0
                    Np[idt(ii[0])]=len[ii]
                idnt=np.unravel_index(idn[:],idt[:],(Mp,L))
                #Quadratic interpolation to better locate the peaks
                a1=TFR.flatten()[idft-1]
                a2=TFR.flatten()[idft]
                a3=TFR.flatten[idft+1]
                dp=(1/2)*(a1-a3)/(a1-2*a2+a3) #check if this operation is working
                #Assign all
                Ip=np.ones((Mp,L))*NaN
                Fp=np.ones((Mp,L))*NaN
                Qp=np.ones((Mp,L))*NaN
                Ip[idnt]=idf+dp
                Qp[idnt]=a2-(1/4)*(a1-a3)*dp 
                if fres==1:
                    Fp[idnt]=freq[idf]+dp[:]*fstep
                else:
                   Fp[idnt]=freq[idf]*np.exp(dp[:]*fstep)
                #Correct "bad" places, if present
                idb=np.argwhere(np.isnan(dp) or np.abs(dp)>1 or idf==1 or idf==NF)
                idb=np.ravel_multi_index(idp,np.shape(dp))
                if idb:
                    Ip[idnt[idb]]=idf[idb]
                    Fp[idnt[idb]]=freq[idf[idb]]
                    Qp[idnt[idb]]=a2[idb]
                #Remove zeros and clear the indices
                TFR=TFR[1:-1,:]
                del idft, idf, idt, idn, dind, idnt, a1, a2, a3, dp
                #Display
                if self.DispMode.lower!='off':
                    if self.DispMode.lower!='notify':
                        print('(number of ridges:', np.round(np.mean(Np[tn1:tn2+1])), ' +- ',np.round(np.std(Np[tn1:tn2+1])), ' from ',np.minimum(Np),' to ',np.maximum(Np),')\n')

                    idp=np.argwhere(Np[tn1:tn2+1]==0)
                    idb=np.ravel_multi_index(idp, np.shape(Np))
                    NB=len(idb);
                    if NB>0:
                       print('Warning: At ', NB, ' times there are no peaks (using border points instead).\n')
                #If there are no peaks, assign border points
                idb=np.ravel_multi_index(np.argwhere(Np[tn1:tn2+1]==0), np.shape(Np)) 
                idb=tn1-1+idb
                NB=len(idb)
                if NB>0:
                    G4=np.abs(TFR[[0,1,NF-1,NF],idb]) #check
                for bn in range(NB):
                    tn=idb[bn]
                    cn=1
                    cg=G4[:,bn]
                    if cg[0]>cg[1] or cg[3]>cg[4]:
                        if cg[0]>cg[1]: 
                            Ip[cn,tn]=1
                            Qp[cn,tn]=cg[0]
                            Fp[cn,tn]=freq[0]
                            cn=cn+1
                        if cg[3]>cg[2]:
                           Ip[cn,tn]=NF
                           Qp[cn,tn]=cg[3]
                           Fp[cn,tn]=freq[NF] 
                           cn=cn+1
                    else:
                        Ip[0:2,tn]=[1,NF]
                        Qp[0:2,tn]=[cg[0],cg[3]]
                        Fp[0:2,tn]=[freq[0],freq[NF]]
                        cn=cn+2
                    
                    Np[tn]=cn-1
                
                del idb, NB, G4
            #if nargout>2, varargout{2}={Np,Ip,Fp,Qp}; end probably not needed
            #substituted with updated skel
            self.Skel['np']=Np
            self.Skel['mt']=Ip
            self.Skel['nu']=Fp
            self.Skel['qn']=Qp
            if self.NormMode.lower=='on':
               nfunc=tfrnormalize(np.abs(TFR[:,tn1:tn2+1]),freq) #defined later
            ci=Ip
            ci[np.isnan(ci)]=NF+2
            cm=ci-np.floor(ci)
            ci=np.floor(ci)
            nfunc=[nfunc[0], nfunc[:], nfunc[-1], NaN,NaN]
            print(np.shape(nfunc))
            Rp=(1-cm)*nfunc[ci+1]+cm*nfunc[ci+2]
            Wp=AmpFunc(Qp*Rp)
            nfunc=nfunc[1:-2] #apply the functional to amplitude peaks
    
        elif type(self.method)!='char' and self.method>3 :#frequency-based extraction
            if len(self.method)!=L:
                error('The specified frequency profile ("Method" property) should be of the same length as signal.')

            efreq=self.method
            submethod=1
            if np.amax(np.abs(efreq.imag))>0:
               submethod=2
               efreq=efreq.imag
            if self.DispMode.lower!='off' and self.DispMode.lower!='notify':
                if submethod==1:
                   print('Extracting the ridge curve lying in the same TFR supports as the specified frequency profile.\n')
                else:
                   print('Extracting the ridge curve lying nearest to the specified frequency profile.\n')

            tn1=np.amax([tn1,np.flatnonzero(np.isnan(efreq)==False)[0]])#we want just the first one
            tn2=np.amin([tn2,np.flatnonzero(np.isnan(efreq)==False)[-1]])
            if fres==1:
               eind=1+np.floor(0.5+(efreq-freq[0])/fstep)
            else:
               eind=1+np.floor(0.5+log(efreq/freq[0])/fstep)
            eind=np.array(eind)
            eind[eind<1]=1
            eind[eind>NF]=NF;
    
            #Extract the indices of the peaks
            for tn in range(tn1,tn2+1):
                cind=eind[tn]
                cs=np.abs(TFR[:,tn])
        
                #Ridge point
                cpeak=cind
                if cind>1 and cind<NF:
                    if cs[cind+1]==cs[cind-1] or submethod==2:
                        cpeak1=cind-1+np.flatnonzero(cs[cind:-1]>=cs[cind-1:-2] and cs[cind:-1]>cs[cind+1:])[0]#check
                        cpeak1=np.amin([cpeak1,NF])
                        cpeak2=cind+1-np.flatnonzero(cs[cind:1:-1]>=cs[cind+1:2:-1] and cs[cind:1:-1]>cs[cind-1::-1])[0]#check
                        cpeak2=np.amax([cpeak2,1])
                        if cs[cpeak1]>0 and cs[cpeak2]>0:
                            if cpeak1-cind==cind-cpeak2:
                                if cs[cpeak1]>cs[cpeak2]:
                                   cpeak=cpeak1
                                else:
                                   cpeak=cpeak2
                            elif cpeak1-cind<cind-cpeak2:
                               cpeak=cpeak1
                            elif cpeak1-cind>cind-cpeak2:
                               cpeak=cpeak2

                        elif cs[cpeak1]==0:
                           cpeak=cpeak2
                        elif cs[cpeak2]==0:
                           cpeak=cpeak1

                    elif cs[cind+1]>cs[cind-1]:
                        cpeak=cind-1+np.flatnonzero(cs[cind:-1]>=cs[cind-1:-1] and cs[cind:-1]>cs[cind+1:])[0]#check
                        cpeak=np.amin([cpeak,NF])
                    elif cs[cind+1]<cs[cind-1]:
                        cpeak=cind+1-np.flatnonzero(cs[cind:0:-1]>cs[cind-1::-1] and cs[cind:0:-1]>=cs[cind+1:1:-1])[0]#check
                        cpeak=np.amax([cpeak,1])
                elif cind==1:
                    if cs[1]<cs[0]:
                       cpeak=cind
                    else:
                        cpeak=1+np.flatnonzero(cs[cind+1:-1]>=cs[cind:-2] and cs[cind+1:-1]>cs[cind+2:])[0]#check
                        cpeak=np.amin([cpeak,NF]);
                elif cind==NF:
                    if cs[NF-1]<cs[NF]:
                       cpeak=cind;
                    else:
                        cpeak=NF-np.flatnonzero(cs[cind-1:0:-1]>cs[cind-2:0:-1] and cs[cind-1::-1]>=cs[cind:1:-1])[0]#check
                        cpeak=np.amax([cpeak,1])

                tfsupp[0,tn]=cpeak;
        
                #Boundaries of time-frequency support
                iup=[]
                idown=[]
                if cpeak<NF-1:
                   iup=cpeak+np.flatnonzero(cs[cpeak+1:-1]<=cs[cpeak:-2] and cs[cpeak+1:-1]<cs[cpeak+2:])[0]

                if cpeak>2:
                   idown=cpeak-np.flatnonzero(cs[cpeak-1:0:-1]<=cs[cpeak:1:-1] and cs[cpeak-1:0:-1]<cs[cpeak-2::-1])[0]#check
                iup=np.amin([iup,NF])
                idown=np.amax([idown,0])
                tfsupp[1,tn]=idown
                tfsupp[2,tn]=iup
    
            #Transform to frequencies
            pind=tfsupp[0,:]
            tfsupp[:,tn1:tn2+1]=freq[tfsupp[:,tn1:tn2+1]]
            pamp[tn1:tn2+1]=np.abs(TFR[np.ravel_multi_index([pind[tn1:tn2+1],np.arange(tn1,tn2+1)],np.shape(TFR))]) #check
    
            # NOT NEEDED?
            ##Optional output arguments
            #if nargout>1
            ec=ec_class(efreq, eind, tfsupp[0,:], pind, pamp, idr)
            #ec.efreq=efreq 
            #ec.eind=eind
            #ec.pfreq=tfsupp[0,:]
            #ec.pind=pind
            #ec.pamp=pamp
            #ec.idr=idr
            #    varargout{1}=ec;
            #end
            #if nargout>2, varargout{2}=[]; end
    
            #Plotting (if needed) #SKIP
            #if ~isempty(strfind(DispMode,'plot'))
            #    scrsz=get(0,'ScreenSize'); figure('Position',[scrsz(3)/4,scrsz(4)/8,2*scrsz(3)/3,2*scrsz(4)/3]);
            #    ax=axes('Position',[0.1,0.1,0.8,0.8],'FontSize',16); hold all;
            #    title(ax(1),'Ridge curve \omega_p(t)/2\pi'); ylabel(ax(1),'Frequency (Hz)'); xlabel(ax(1),'Time (s)');
            #    plot(ax(1),(0:L-1)/fs,efreq,'--','Color',[0.5,0.5,0.5],'LineWidth',2,'DisplayName','Specified frequency profile');
            #    plot(ax(1),(0:L-1)/fs,tfsupp(1,:),'-k','LineWidth',2,'DisplayName','Extracted frequency profile');
            #    legend(ax(1),'show'); if fres==2, set(ax(1),'YScale','log'); end
            #end
            #if ~isempty(strfind(PlotMode,'on')), plotfinal(tfsupp,TFR,freq,fs,DispMode,PlotMode); end
            #if nargout>2, varargout{2}=Skel; end

            self.Skel['Np']=Np
            self.Skel['mt']=Ip
            self.Skel['nu']=Fp
            self.Skel['qn']=Qp
    
            return tfsupp,ecinfo, ec, self.Skel

        #--------------------------- Global Maximum -------------------------------
        if (type(self.method) is 'char' and self.method.lower=='max') or len(self.pars)==2:
            if self.DispMode.lower!='off' and self.DispMode!='notify':
                if sflag==0:
                   print('Extracting the curve by Global Maximum scheme.\n')
                else:
                   print('Extracting positions of global maximums (needed to estimate initial parameters).\n')
    
            if sflag==0:
                if self.NormMode.lower=='on':
                    nfunc=tfrnormalize(np.abs(TFR[:,tn1:tn2+1]),freq)
                    TFR=TFR*(nfunc[:]@ones((1,L))) #this should be a vector moltiplication (not element-wise)

                for tn in range(tn1,tn2+1):
                   [pamp[tn],pind[tn]]=np.unravel_index(np.argmax(abs(TFR[:,tn])),np.shape(TFR[:,tn]))
                tfsupp[0,tn1:tn2+1]=freq[pind[tn1:tn2+1]]
                if self.NormMode.lower=='on':
                    TFR=TFR/(nfunc[:]@ones((1,L)))
                    pamp[tn1:tn2+1]=pamp[tn1:tn2+1]/(nfunc[pind[tn1:tn2+1]].T)
            else:
                for tn in range(tn1,tn2+1):
                   [pamp[tn],idr[tn]]=np.unravel_index(np.argmax(Wp[0:Np[tn]+1,tn]), np.shape(Wp[0:Np[tn]+1,tn]))
                
                lid=ravel_multi_index([idr[tn1:tn2+1],np.arange(tn1,tn2+1)],np.shape(Fp)) 
                tfsupp[0,tn1:tn2+1]=Fp[lid]
                pind[tn1:tn2+1]=np.round[Ip[lid]]
                pamp[tn1:tn2+1]=Qp[lid]
            
            idz=tn1-1+np.flatnonzero(pamp[tn1:tn2+1 ]==0 or np.isnan(pamp[tn1:tn2+1]))
            if idz:
                idnz=np.arange(tn1,tn2+1)
                idnz=idnz(np.argwhere(np.isin(idnz,idz)==False)) # =>np.in1d

                pind[idz]=np.interp(idz,idnz,pind[idnz]) #'linear' in the function no equivalent of 'extrap'
                pind[idz]=np.round(pind[idz])
                tfsupp[0,idz]=np.interp(idz,idnz,tfsupp[0,idnz]) #'linear' in the function no equivalent of 'extrap'
 
            #if nargout>1, ec.pfreq=tfsupp(1,:); ec.pind=pind; ec.pamp=pamp; ec.idr=idr; end
            #if ~isempty(strfind(DispMode,'plot')) && strcmpi(method,'max')
            #    scrsz=get(0,'ScreenSize'); figure('Position',[scrsz(3)/4,scrsz(4)/8,2*scrsz(3)/3,2*scrsz(4)/3]);
            #    ax=axes('Position',[0.1,0.1,0.8,0.8],'FontSize',16); hold all;
            #    title(ax(1),'Ridge curve \omega_p(t)/2\pi'); ylabel(ax(1),'Frequency (Hz)'); xlabel(ax(1),'Time (s)');
            #    plot(ax(1),(0:L-1)/fs,tfsupp(1,:),'-k','LineWidth',2,'DisplayName','Global Maximum curve');
            #    legend(ax(1),'show'); if fres==2, set(ax(1),'YScale','log'); end
         


        #------------------------- Nearest neighbour ------------------------------
        if (type(self.method) is 'char') and self.method.lower=='nearest':
            #Display, if needed
            imax=np.argmax(Wp.flatten());
            [fimax,timax]=np.unravel_index(imax,(Mp,L)) 
            idr[timax]=fimax
            if self.DispMode.lower!='off' and self.DispMode.lower!='notify':
                print('Extracting the curve by Nearest Neighbour scheme.\n')
                print('The highest peak was found at time ', (timax-1)/fs,  ' s and frequency', Fp(fimax,timax), ' Hz (indices ', timax,' and ',Ip(fimax,timax),' respectively).\n')
                print('Tracing the curve forward and backward from point of maximum.\n')
            #Main part
            for tn in range(timax+1,tn2+1):
               idr[tn]=np.argmin(np.abs(Ip[0:Np[tn],tn]-idr[tn-1]))
            for tn in range(timax-1,tn1-1,-1):
               idr[tn]=np.argmin(np.abs(Ip[0:Np[tn],tn]-idr[tn+1]))
            lid=np.ravel_multi_index(idr[tn1:tn2],np.arange(tn1,tn2+1),np.shape(Fp))
            tfsupp[0,tn1:tn2+1]=Fp[lid]
            pind[tn1:tn2+1]=np.round(Ip[lid])
            pamp[tn1:tn2+1]=Qp[lid]
            #Assign the output structure and display, if needed
            #if nargout>1
            ec=ec_class(pfreq=tfsupp[0,:], pind=pind, pamp=pamp, idr=idr)
            #ec.pfreq=tfsupp[0,:]
            #ec.pind=pind
            #ec.pamp=pamp 
            #ec.idr=idr
            #if ~isempty(strfind(DispMode,'plot'))
            #    scrsz=get(0,'ScreenSize'); figure('Position',[scrsz(3)/4,scrsz(4)/8,2*scrsz(3)/3,2*scrsz(4)/3]);
            #    ax=axes('Position',[0.1,0.1,0.8,0.8],'FontSize',16); hold all;
            #    title(ax(1),'Ridge curve \omega_p(t)/2\pi'); ylabel(ax(1),'Frequency (Hz)'); xlabel(ax(1),'Time (s)');
            #    plot(ax(1),(0:L-1)/fs,tfsupp(1,:),'-k','LineWidth',2,'DisplayName','Nearest Neighbour curve');
            #    plot(ax(1),(timax-1)/fs,Fp(fimax,timax),'ob','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','r',...
            #        'DisplayName','Starting point (overall maximum of TFR amplitude)');
            #    legend(ax(1),'show'); if fres==2, set(ax(1),'YScale','log'); end
            

        #----------------------------- Method I -----------------------------------
        if len(self.pars)==1:
            if self.DispMode.lower!='off' and self.DispMode.lower!='notify':
                print('Extracting the curve by I scheme.\n')
            #Define the functionals
            #if not PenalFunc[1]:
            #    def logw1(x):
            #        return (x)-pars(1)*abs(fs*x/DD)
            #else:
            #   def logw1(x):
            #       return (x)*PenalFunc[1][fs*x,DD]
            #if not PenalFunc[2]:
            #   def logw2(x):
            #       return []
            #else:
            #   def logw2(x):
            #       return (x)*PenalFunc[2][x,DF]

            if not self.PenalFunc['1']:
                logw1= lambda x: (x)-self.pars(1)*abs(fs*x/DD)
            else:
               logw1= lambda x: (x)*self.PenalFunc['1'][fs*x,DD] #check
            if not self.PenalFunc['2']:
               logw2= lambda x: []
            else:
               logw2=lambda x: (x)*self.PenalFunc['2'][x,DF] #check
            #Main part
            if PathOpt.lower=='on':
                idr=pathopt(Np,Ip,Fp,Wp,logw1,logw2,freq,self.DispMode) #check
            else:
               [idr,timax,fimax]=onestepopt(Np,Ip,Fp,Wp,logw1,logw2,freq,self.DispMode)
            lid=np.ravel_multi_index(idr[tn1:tn2+1],np.arange[tn1:tn2+1], np.shape(Fp))
            tfsupp[0,tn1:tn2+1]=Fp[lid]
            pind[tn1:tn2+1]=np.round(Ip[lid])
            pamp[tn1:tn2+1]=Qp[lid]

            #Assign the output structure and display, if needed
        #    if nargout>1, 
            ec=ec_class(pfreq=tfsupp[0,:], pind=pind, pamp=pamp, idr=idr)
        #    ec.pfreq=tfsupp[0,:]
        #    ec.pind=pind
        #    ec.pamp=pamp
        #    ec.idr=idr
        ##    if ~isempty(strfind(DispMode,'plot'))
        #        scrsz=get(0,'ScreenSize'); figure('Position',[scrsz(3)/4,scrsz(4)/8,2*scrsz(3)/3,2*scrsz(4)/3]);
        #        ax=axes('Position',[0.1,0.1,0.8,0.8],'FontSize',16); hold all;
        #        title(ax(1),'Ridge curve \omega_p(t)/2\pi'); ylabel(ax(1),'Frequency (Hz)'); xlabel(ax(1),'Time (s)');
        #        plot(ax(1),(0:L-1)/fs,tfsupp(1,:),'-k','LineWidth',2,'DisplayName','Extracted frequency profile');
        #        if ~strcmpi(PathOpt,'on')
        #            plot(ax(1),(timax-1)/fs,Fp(fimax,timax),'ob','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','r',...
        #                'DisplayName','Starting point (overall maximum of local functional)');
        #        end
        #        legend(ax(1),'show'); if fres==2, set(ax(1),'YScale','log'); end
        #    end
        #end


        #----------------------------- Method II ----------------------------------
        if len(self.pars)==2 and self.method==2:
            #Initialize the parameters
            pf=tfsupp[0,tn1:tn2+1]
            if fres==2:
               pf=np.log(pf)
            dpf=np.diff(pf,1,0)
            mv=[np.median(dpf),0,np.median(pf),0];
            ss1=np.sort(dpf)
            CL=len(ss1)
            mv[1]=ss1[np.round(0.75*CL)]-ss1[np.round(0.25*CL)]
            ss2=np.sort(pf)
            CL=len(ss2)
            mv[3]=ss2[np.round(0.75*CL)]-ss2[np.round(0.25*CL)]
    
            #Display, if needed
            if self.DispMode.lower!='off' and self.DispMode!='notify':
                if fres==1:
                    print(['maximums frequencies (median+-range): '])
                    print(mv[2],'+-',mv[3],' hz; frequency differences: ',mv[0],'+-',mv[1] ,'hz.\n')
                else:
                    print(['maximums frequencies (log-median*/range ratio): ']);
                    fprintf(exp(mv[2]),'*/',exp(mv[3]), 'hz; frequency ratios:',exp(mv[0]),'*/',exp(mv[1]),'\n')
                print('extracting the curve by ii scheme: iteration discrepancy - ');
            #    if ~isempty(strfind(DispMode,'plot'))
            #        scrsz=get(0,'ScreenSize'); figure('Position',[scrsz(3)/4,scrsz(4)/8,2*scrsz(3)/3,2*scrsz(4)/3]);
            #        ax=zeros(3,1);
            #        ax(1)=axes('Position',[0.1,0.6,0.8,0.3],'FontSize',16); hold all;
            #        if fres==2, set(ax(1),'YScale','log'); end
            #        ax(2)=axes('Position',[0.1,0.1,0.35,0.35],'FontSize',16); hold all;
            #        ax(3)=axes('Position',[0.55,0.1,0.35,0.35],'FontSize',16); hold all;
            #        title(ax(1),'Ridge curve \omega_p(t)/2\pi'); ylabel(ax(1),'Frequency (Hz)'); xlabel(ax(1),'Time (s)');
            #        ylabel(ax(2),'Frequency (Hz)'); ylabel(ax(3),'Frequency (Hz)'); xlabel(ax(3),'Iteration number'); xlabel(ax(2),'Iteration number');
            #        title(ax(2),'${\rm m}[d\omega_p/dt]/2\pi$ (solid), ${\rm s}[d\omega_p/dt]/2\pi$ (dashed)','interpreter','Latex','FontSize',20);
            #        title(ax(3),'${\rm m}[\omega_p]/2\pi$ (solid), ${\rm s}[\omega_p]/2\pi$ (dashed)','interpreter','Latex','FontSize',20);
            #        line0=plot(ax(1),(0:L-1)/fs,tfsupp(1,:),':','Color',[0.5,0.5,0.5],'DisplayName','Global Maximum ridges');
            #        line1=plot(ax(2),0,fs*mv(1),'-sk','LineWidth',2,'MarkerSize',6,'MarkerFaceColor','k','DisplayName','m[d\omega_p/dt]/2\pi');
            #        line2=plot(ax(2),0,fs*mv(2),'--ok','LineWidth',2,'MarkerSize',6,'MarkerFaceColor','k','DisplayName','s[d\omega_p/dt]/2\pi');
            #        line3=plot(ax(3),0,mv(3),'-sk','LineWidth',2,'MarkerSize',6,'MarkerFaceColor','k','DisplayName','m[\omega_p]/2\pi');
            #        line4=plot(ax(3),0,mv(4),'--ok','LineWidth',2,'MarkerSize',6,'MarkerFaceColor','k','DisplayName','s[\omega_p]/2\pi');
            #    end
            #end
    
            #Iterate
            rdiff=NaN
            itn=0
            allpind=np.zeros((10,L))
            allpind[0,:]=pind
            ec=ec_class() #hope that this will work.
            ec.mv=mv
            ec.rdiff=rdiff
            while rdiff!=0:
                #Define the functionals
                smv=[mv[1],mv[3]] #to avoid underflow
                if smv[0]<=0:
                   smv[0]=10**(-32)/fs
                if smv[1]<=0:
                    smv[1]=10**(-16)
                if not self.PenalFunc['1']:
                   logw1= lambda x: (x)-self.pars[1]*abs((x-mv(1))/smv(1))
                else:
                   logw1= lambda x: (x)*self.PenalFunc['1'][x,mv(1),smv(1)] 
                if not self.PenalFunc['2']:
                   logw2 = lambda x: (x)-self.pars[1]*abs((x-mv[2])/smv[1]) 
                else:
                   logw2 = lambda x: (x)*self.PenalFunc['2'][x,mv[2],smv[1]]
                #Calculate all
                pind0=pind;
                if PathOpt.lower=='on':
                   idr=pathopt(Np,Ip,Fp,Wp,logw1,logw2,freq,self.DispMode) #check
                else:
                   idr, timax, fimax=onestepopt(Np,Ip,Fp,Wp,logw1,logw2,freq,self.DispMode) #check
                lid=np.ravel_multi_index(idr[tn1:tn2+1],np.arange(tn1,tn2+1),np.shape(Fp))
                tfsupp[0,tn1:tn2+1]=Fp[lid]
                pind[tn1:tn2+1]=np.round(Ip[lid]) 
                pamp[tn1:tn2+1]=Qp[lid]
                rdiff=len(np.ravel_multi_index(np.nonzero(pind[tn1:tn2+1]-pind0[tn1:tn2+1]!=0,np.shape(pind))))/(tn2-tn1+1)
                itn=itn+1;
                #Update the medians/ranges
                pf=tfsupp[0,tn1:tn2+1] 
                if fres==2:
                   pf=log(pf)
                dpf=np.diff(pf,1,0)
                mv=[np.median(dpf),0,np.median(pf),0]
                ss1=np.sort(dpf)
                CL=len(ss1)
                mv[1]=ss1[np.round(0.75*CL)]-ss1[np.round(0.25*CL)]
                ss2=np.sort(pf)
                CL=len(ss2)
                mv[3]=ss2[np.round(0.75*CL)]-ss2[round(0.25*CL)]
                #Update the information structure, if needed
                
                ec.pfreq.append(tfsupp[0,:])
                ec.pind.append(pind)
                ec.pamp.append(pamp)
                ec.idr.append(idr)
                ec.mv.append(mv)
                ec.rdiff.append(rdiff)
                
                ##Display, if needed
                #if ~strcmpi(DispMode,'off') && ~strcmpi(DispMode,'notify')
                #    fprintf('%0.2f%%; ',100*rdiff);
                #    if ~isempty(strfind(DispMode,'plot'))
                #        line0=plot(ax(1),(0:L-1)/fs,tfsupp(1,:),'DisplayName',sprintf('Iteration %d (discrepancy %0.2f%%)',itn,100*rdiff));
                #        set(line1,'XData',0:itn,'YData',[get(line1,'YData'),fs*mv(1)]);
                #        set(line2,'XData',0:itn,'YData',[get(line2,'YData'),fs*mv(2)]);
                #        set(line3,'XData',0:itn,'YData',[get(line3,'YData'),mv(3)]);
                #        set(line4,'XData',0:itn,'YData',[get(line4,'YData'),mv(4)]);
                #        if ~strcmpi(PathOpt,'on'),
                #            mpt=plot(ax(1),(timax-1)/fs,Fp(fimax,timax),'ok','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','w',...
                #                'DisplayName',['Starting point (iteration ',num2str(itn),')']);
                #        end
                #    end
                #end

                #Stop if maximum number of iterations has been reached
                if itn>MaxIter and rdiff!=0:
                    if self.DispMode.lower!='off':
                        if self.DispMode.lower!='notify':
                           print('\n')
                        print('WARNING! Did not fully converge in %d iterations (current ''MaxIter''). Using the last estimate.',MaxIter);
                    break
                
                #Just in case, check for ``cycling'' (does not seem to occur in practice)
                allpind[itn+1,:]=pind
                gg=math.inf
                if rdiff!=0 and itn>2:
                    for kn in range(1,itn):
                       gg=np.amin([gg,len(np.ravel_multi_index(np.nonzero(pind[tn1:tn2+1]-allpind[kn,tn1:tn2+1]!=0)))])
                
                if gg==0:
                    if self.DispMode.lower!='off' and self.DispMode.lower!='notify':
                        print('converged to a cycle, terminating iteration.')
                    
                    break
  
            if self.DispMode.lower!='off' and self.DispMode.lower!='notify':
               print('\n')
        #    if ~isempty(strfind(DispMode,'plot'))
        #        set(line0,'Color','k','LineWidth',2);
        #        if ~strcmpi(PathOpt,'on'), set(mpt,'Color',b,'MarkerFaceColor','k'); end
        #        if fres==2 %change plot if the resolution is logarithmic
        #            set(line1,'YData',exp(get(line1,'YData')),'DisplayName','exp(m[d\log\omega_p/dt])');
        #            set(line2,'YData',exp(get(line2,'YData'))-1,'DisplayName','exp(s[d\log\omega_p/dt])-1');
        #            set(line3,'YData',exp(get(line3,'YData')),'DisplayName','exp(m[\log\omega_p])/2\pi');
        #            set(line4,'YData',exp(get(line4,'YData'))-1,'DisplayName','exp(s[\log\omega_p])-1');
        #            set(ax(2:3),'YScale','log'); ylabel(ax(2),'Frequency Ratio'); ylabel(ax(3),'Frequency (Hz)');
        #            title(ax(2),'$e^{{\rm m}[d\log\omega_p/dt]}$ (solid), $e^{{\rm s}[d\log\omega_p/dt]}-1$ (dashed)','interpreter','Latex','FontSize',20);
        #            title(ax(3),'$e^{{\rm m}[\log\omega_p]}/2\pi$ (solid), $e^{{\rm s}[\log\omega_p]}-1$ (dashed)','interpreter','Latex','FontSize',20);
        #            iline=[get(line1,'YData'),get(line2,'YData')]; set(ax(2),'YLim',[0.75*min(iline),1.5*max(iline)]);
        #            iline=[get(line3,'YData'),get(line4,'YData')]; set(ax(3),'YLim',[0.75*min(iline),1.5*max(iline)]);
        #        end
        #    end
    
        #end
        #varargout[1]=ec #check

        #//////////////////////////////////////////////////////////////////////////
        #Extract the time-frequency support around the ridge points
        for tn in range(tn1,tn2+1):
            cs=np.abs(TFR[:,tn]) 
            cpeak=pind[tn]
            iup=[]
            idown=[]
            if cpeak<NF-1:
               iup=cpeak+np.ravel_multi_index(np.nonzero(cs[cpeak+1:-1]<=cs[cpeak:-2] and cs[cpeak+1:-1]<cs[cpeak+2:]))[0]
            if cpeak>2: 
                idown=cpeak-np.ravel_multi_index(np.nonzero(cs[cpeak-1:0:-1]<=cs[cpeak:1:-1] and cs[cpeak-1:0:-1]<cs[cpeak-2::-1]))
            iup=np.amin(iup,NF) 
            idown=np.amax(idown,1)
            tfsupp[1,tn]=idown
            tfsupp[2,tn]=iup
 
        tfsupp[1:,tn1:tn2+1]=freq[tfsupp[1:,tn1:tn2+1]]
        #//////////////////////////////////////////////////////////////////////////

        #Final display
        if self.DispMode.lower!='off' and self.DispMode.lower!='notify':
            print('Curve extracted: ridge frequency ',np.mean(tfsupp[0,:]),'+-',np.std(tfsupp[0,:]),' Hz, lower support ', np.mean(tfsupp[1,:]),'+-',np.std(tfsupp[1,:]) ,' Hz, upper support ',np.mean(tfsupp[2,:]),'+-',np.std(tfsupp[2,:]),' Hz.\n')
        

        ##Plot (if needed)
        #if ~isempty(strfind(PlotMode,'on')), plotfinal(tfsupp,TFR,freq,fs,DispMode,PlotMode,nfunc); end

        #end

        self.Skel['Np']=Np
        self.Skel['mt']=Ip
        self.Skel['nu']=Fp
        self.Skel['qn']=Qp

        return tfsupp,ecinfo,self.Skel, ec

        #==========================================================================
        #========================= Support functions ==============================
        #==========================================================================

        #==================== Path optimization algorithm =========================
    def pathopt(self,Np,Ip,Fp,Wp,logw1, logw2,freq):
        #maybe logw1 and logw2 not neeeded
        DispMode=self.DispMode
        [Mp,L]=np.shape(Fp)
        NF=len(freq)
        tn1=np.ravel_multi_index(np.nonzero(Np>0))[0] 
        tn2=np.ravel_multi_index(np.nonzero(Np>0))[-1]
        if np.amin(freq)>0 and np.std(np.diff(freq,1,0))>np.std(np.diff(np.log(freq),1,0)): 
            Fp=np.log(Fp)

        #Weighting functions
        if not isinstance(logw1, types.LambdaType):
            if logw1==[]:
                logw1=np.zeros((2*NF+1,L))
            else:
                logw1=[[2*logw1[0]-logw1[1]],[logw1[:]],[2*logw1[-1]-logw1[-2]]] #from here
        #    end
        #end
        if not isinstance(logw2, types.LambdaType):
            if logw2==[]:
                W2=np.zeros((Mp,L))*NaN
            else:
                logw2=[[2*logw2[0]-logw2[1]],[logw2[:]],[2*logw2[-1]-logw2[-2]],[NaN],[NaN]]
                ci=Ip
                ci[np.isnan(ci)]=NF+2
                cm=ci-np.floor(ci)
                ci=np.floor(ci);
                W2=(1-cm)*logw2[ci+1]+cm*logw2[ci+2]
                del ci, cm
            
        else:
            W2=logw2(Fp)
        
        W2=Wp+W2;

        #The algorithm by itself
        q=np.zeros((Mp,L))*NaN 
        U=np.zeros((Mp,L))*NaN
        q[0:Np[tn1]+1,tn1]=0
        U[0:Np[tn1]+1,tn1]=W2[0:Np[tn1]+1,tn1]


        #TODO: THIS
        if isinstance(logw1, types.LambdaType):
            for tn in range(tn1+1,tn2+1):
                cf=Fp[0:Np[tn]+1,tn]*np.ones((1,Np[tn-1]))-np.ones((Np[tn]+1,1))*Fp[0:Np[tn-1]+1,tn-1].T
                CW1=logw1(cf);
                [U[0:Np[tn]+1,tn],q[0:Np[tn]+1,tn]]=np.amax(W2[0:Np[tn]+1,tn]*np.ones((1,Np[tn-1]))+CW1+np.ones((Np[tn],1))*U[0:Np[tn-1]+1,tn-1].T,1)#max along  rows
        else:
            for tn in range(tn1+1,tn2+1):
                ci=Ip[0:Np[tn]+1,tn]*np.ones((1,Np[tn-1]))-np.ones((Np[tn],1))*Ip[0:Np[tn-1]+1,tn-1].T
                ci=ci+NF
                cm=ci-np.floor(ci)
                ci=np.floor(ci)
                if Np(tn)>1:
                    CW1=(1-cm)*logw1(ci+1)+cm*logw1(ci+2)
                else:
                   CW1=(1-cm)*logw1(ci+1).T+cm*logw1(ci+2).T
                [U[0:Np[tn]+1,tn],q[0:Np[tn]+1,tn]]=np.amax(W2[0:Np[tn]+1,tn]*np.ones((1,Np[tn-1]))+CW1+np.ones((Np[tn],1))*U[0:Np[tn-1]+1,tn-1].T,1)#max along  rows

        #Recover the indices
        idid=np.zeros((1,L))*NaN
        idid[tn2]=np.argmax(U[:,tn2])
        for tn in np.arange(tn2-1,tn1+1,-1):
           idid[tn]=q[idid[tn+1],tn+1]

        return idid

        #%Plot if needed
        #%{
        #if ~isempty(strfind(DispMode,'plot+'))
        #    figure; axes('FontSize',16,'Box','on'); hold all;
        #    rlines=zeros(Np(tn2),1);
        #    for pn=1:Np(tn2)
        #        cridges=zeros(L,1)*NaN; cridges(tn2)=pn;
        #        for tn=tn2-1:-1:tn1
        #            cridges(tn)=q(cridges(tn+1),tn+1);
        #        end
        #        lind=sub2ind(size(Ip),cridges(tn1:tn2),(tn1:tn2)');
        #        cridges=[ones(tn1-1,1)*NaN;Ip(lind);ones(L-tn2,1)*NaN];
        #        crfreq=[ones(tn1-1,1);freq(cridges(tn1:tn2));ones(L-tn2,1)];
        #        rlines(pn)=plot((0:L-1)/fs,crfreq,'DisplayName',['U=',num2str(U(pn,tn2))]);
        #    end
        #    [~,imax]=max(U(:,tn2)); uistack(rlines(imax),'top');
        #    set(rlines(imax),'Color','k','LineWidth',2);
        #    title('Path to each of the last peaks maximizing the given functional');
        #    ylabel('Frequency (Hz)'); xlabel('Time (s)');
        #end
        #%}

        

        #================== One-step optimization algorithm =======================
    def onestepopt(self,Np,Ip,Fp,Wp,freq):

         #maybe logw1 and logw2 not neeeded
        DispMode=self.DispMode
        [Mp,L]=np.shape(Fp)
        NF=len(freq)
        tn1=np.ravel_multi_index(np.nonzero(Np>0))[0]
        tn2=np.ravel_multi_index(np.nonzero(Np>0))[-1]
        if np.amin(freq)>0 and np.std(np.diff(freq,1,0))>np.std(np.diff(np.log(freq),1,0)):
           Fp=np.log(Fp)


           ##TO DO
        #%Weighting functions
        if not isinstance(logw1, types.LambdaType):
            if logw1==[]:
                logw1=np.zeros((2*NF+1,L))
            else:
                logw1=[[2*logw1[0]-logw1[1]],[logw1[:]],[2*logw1[-1]-logw1[-2]]]

        if not isinstance(logw2, types.LambdaType):
            if logw2==[]:
                W2=np.zeros((Mp,L))
            else:
                logw2=[[2*logw2[0]-logw2[1]],[logw2[:]],[2*logw2[-1]-logw2[-2]],[NaN],[NaN]];
                ci=Ip
                ci[np.isnan(ci)]=NF+2
                cm=ci-np.floor(ci)
                ci=np.floor(ci)
                W2=(1-cm)*logw2[ci+1]+cm*logw2[ci+2]
                del ci, cm
            
        else:
            W2=logw2(Fp)
        
        W2=Wp+W2

        #The algorithm by itself
        imax=np.argmax(W2.flatten()[:])
        fimax,timax=np.unravel_index(imax,[Mp,L]) #determine the starting point
        idid=np.zeros((1,L))*NaN
        idid[timax]=fimax

        #TO DO
        if isinstance(logw1, types.LambdaType):
            for tn in range(timax+1,tn2+1):
                cf=Fp[1:Np[tn],tn]-Fp[idid[tn-1],tn-1]
                CW1=logw1(cf)
                idid[tn]=np.unravel_index(np.argmax(W2[0:Np[tn]+1,tn]+CW1),np.shape(W2))[1]
            for tn in range(timax-1,tn1-1,-1):
                cf=-Fp[0:Np[tn]+1,tn]+Fp[idid[tn+1],tn+1]
                CW1=logw1(cf)
                idid[tn]=np.unravel_index(np.argmax(W2[0:Np[tn]+1,tn]+CW1),np.shape(W2))[1]
            
        else:
            for tn in range(timax+1,tn2+1):
                ci=NF+Ip[0:Np[tn]+1,tn]-Ip[idid[tn-1],tn-1]
                cm=ci-np.floor(ci)
                ci=np.floor(ci);
                CW1=(1-cm)*logw1[ci+1]+cm*logw1[ci+2]
                idid[tn]=np.unravel_index(np.argmax(W2[0:Np(tn)+1,tn]+CW1),np.shape(W2))
            
            for tn in range(timax-1,tn-1,-1):
                ci=NF-Ip[0:Np(tn)+1,tn]+Ip[idid[tn+1],tn+1]
                cm=ci-np.floor(ci)
                ci=np.floor[ci]
                CW1=(1-cm)*logw1[ci+1]+cm*logw1[ci+2] 
                idid[tn]=np.unravel_index(np.argmax(W2[0:Np[tn]+1,tn]+CW1),np.shape(W2))
     

        ##if nargout>1, 
        #varargout{1}=timax
        ##if nargout>2, 
        #varargout{2}=fimax

        return idid, timax, fimax

        #========================= Plotting function ==============================
        #function plotfinal(tfsupp,TFR,freq,fs,DispMode,PlotMode,varargin)

        #[NF,L]=size(TFR); fres=1; if min(freq)>0 && std(diff(freq))>std(diff(log(freq))), fres=2; end
        #nfunc=ones(NF,1); if nargin>6 && ~isempty(varargin{1}), nfunc=varargin{1}; end
        #XX=(0:(L-1))/fs; YY=freq; ZZ=abs(TFR).*(nfunc(:)*ones(1,L)); scrsz=get(0,'ScreenSize');

        #MYL=round(scrsz(3)); MXL=round(scrsz(4)); %maximum number of points seen in plots
        #if isempty(strfind(lower(PlotMode),'wr')) && (size(ZZ,1)>MYL || size(ZZ,2)>MXL)
        #    if ~strcmpi(DispMode,'off') && ~strcmpi(DispMode,'notify')
        #        fprintf('Plotting: TFR contains more data points (%d x %d) than pixels in the plot, so for a\n',size(ZZ,1),size(ZZ,2));
        #        fprintf('          better performance its resampled version (%d x %d) will be displayed instead.\n',min([MYL,size(ZZ,1)]),min([MXL,size(ZZ,2)]));
        #    end
        #    XI=XX; YI=YY;
        #    if size(ZZ,2)>MXL, XI=linspace(XX(1),XX(end),MXL); end
        #    if fres==1
        #        if size(ZZ,1)>MYL, YI=linspace(YY(1),YY(end),MYL); end
        #        ZZ=aminterp(XX,YY,ZZ,XI,YI,'max');
        #    else
        #        if size(ZZ,1)>MYL, YI=exp(linspace(log(YY(1)),log(YY(end)),MYL)); end
        #        ZZ=aminterp(XX,log(YY),ZZ,XI,log(YI),'max');
        #    end
        #    XX=XI(:); YY=YI(:);
        #end

        #figure('Position',[scrsz(3)/6,scrsz(4)/4,2*scrsz(3)/3,2*scrsz(4)/3]);
        #ax0=axes('Position',[0.1,0.15,0.8,0.7],'NextPlot','Add','FontSize',18,...
        #    'XLim',[XX(1),XX(end)],'YLim',[YY(1),YY(end)],'Layer','top','Box','on');
        #xlabel('Time (s)'); ylabel('Frequency (Hz)');
        #title({'TFR amplitude', '(black: extracted peaks; gray: their supports)'});
        #if fres==2, set(gca,'YScale','log'); end
        #pc=pcolor(XX,YY,ZZ); set(pc,'LineStyle','none');
        #plot((0:(L-1))/fs,tfsupp(1,:),'Color','k','LineWidth',2);
        #plot((0:(L-1))/fs,tfsupp(2,:),'Color',[0.5,0.5,0.5],'LineWidth',2);
        #plot((0:(L-1))/fs,tfsupp(3,:),'Color',[0.5,0.5,0.5],'LineWidth',2);
        #if ~isempty(find(nfunc~=1,1))
        #    set(ax0,'Position',[0.1,0.15,0.65,0.7]);
        #    title(ax0,{'Normalized TFR amplitude', '(black: extracted peaks; gray: their supports)'});
        #    axes('Position',[0.8,0.15,0.175,0.7],'FontSize',18,'XLim',[0,1.25],'YLim',[freq(1),freq(end)],'Box','on');
        #    hold all; plot(nfunc,freq,'-k','LineWidth',2); title({'Normalization','function'});
        #    set(gca,'XTick',[0,0.5,1],'YTickLabel',{}); if fres==2, set(gca,'YScale','log'); end
        #end

        #end

        #%=============== Interpolation function for plotting ======================
        #% ZI=aminterp(X,Y,Z,XI,YI,method)
        #% - the same as interp2, but uses different interpolation types,
        #% maximum-based ([method]='max') or average-based ([method]='avg'), where
        #% [ZI] at each point [XI,YI] represents maximum or average among values of
        #% [Z] corresponding to the respective quadrant in [X,Y]-space which
        #% includes point [XI,YI];
        #% X,Y,XI,YI are all linearly spaced vectors, and the lengths of XI,YI
        #% should be the same or smaller than that of X,Y; Z should be real,
        #% ideally positive; X and Y correspond to the 2 and 1 dimension of Z, as
        #% always.
        #%--------------------------------------------------------------------------
        #function ZI = aminterp(X,Y,Z,XI,YI,method)

        #%Interpolation over X
        #ZI=zeros(size(Z,1),length(XI))*NaN;
        #xstep=mean(diff(XI));
        #xind=1+floor((1/2)+(X-XI(1))/xstep); xind=xind(:);
        #xpnt=[0;find(xind(2:end)>xind(1:end-1));length(xind)];
        #if strcmpi(method,'max')
        #    for xn=1:length(xpnt)-1
        #        xid1=xpnt(xn)+1; xid2=xpnt(xn+1);
        #        ZI(:,xind(xid1))=max(Z(:,xid1:xid2),[],2);
        #    end
        #else
        #    for xn=1:length(xpnt)-1
        #        xid1=xpnt(xn)+1; xid2=xpnt(xn+1);
        #        ZI(:,xind(xid1))=mean(Z(:,xid1:xid2),2);
        #    end
        #end

        #Z=ZI;

        #%Interpolation over Y
        #ZI=zeros(length(YI),size(Z,2))*NaN;
        #ystep=mean(diff(YI));
        #yind=1+floor((1/2)+(Y-YI(1))/ystep); yind=yind(:);
        #ypnt=[0;find(yind(2:end)>yind(1:end-1));length(yind)];
        #if strcmpi(method,'max')
        #    for yn=1:length(ypnt)-1
        #        yid1=ypnt(yn)+1; yid2=ypnt(yn+1);
        #        ZI(yind(yid1),:)=max(Z(yid1:yid2,:),[],1);
        #    end
        #else
        #    for yn=1:length(ypnt)-1
        #        yid1=ypnt(yn)+1; yid2=ypnt(yn+1);
        #        ZI(yind(yid1),:)=mean(Z(yid1:yid2,:),1);
        #    end
        #end

        #end

        #================= Normalization of the noise peaks =======================
    def tfrnormalize(self,TFR,freq):

        #Calculate medians and percentiles
        NF=len(freq)
        L=np.shape(TFR)[1] 
        TFR=np.sort(TFR,1)
        mm=np.median(TFR,1)
        pp=0.75
        ss=TFR[:,np.round((0.5+pp/2)*L)]-TFR[:,np.round((0.5-pp/2)*L)]

        #Calculate weightings
        gg=ss/mm
        gg[np.isnan(gg)]=math.Inf #check this could not work
        ii=np.ravel_multi_index(np.nonzero(np.isfinite(gg))) 
        gg=gg-np.median(gg[ii])
        zz=np.sort(np.abs(gg[ii]))
        zz=zz(np.round(0.25*len(zz))) 
        rw=np.exp(-np.abs(gg)/zz/2)
        rw=np.ones((NF,1))

        #Fitting
        Y=mm
        ii=np.ravel_multi_index(np.nonzero(freq>0 and Y>0))
        CN=len(ii)
        Y=rw[ii]*log(Y[ii])
        X=np.log(freq[ii])
        FM=(rw[ii]@np.ones((1,2)))@[np.ones((CN,1)),X[:]]
        b=pinv[FM]*Y[:]

        #Construct the normalization function
        nfunc=freq[freq>0]**b[1] 
        nfunc=[nfunc[0]*np.ones(len(freq[freq<=0]),1),[nfunc[:]]]
        dd=mm[:]-np.exp(b[0])*nfunc
        pid=np.ravel_multi_index(np.nonzero(dd>0 and freq>0))
        rid=np.argmin(np.abs(np.cumsum(dd[pid])-0.5*np.sum(dd[pid])))
        mff=freq[pid[rid]]
        nfunc=(mff**b[1])/nfunc
        nfunc[nfunc>1 or np.isnan(nfunc)]=1 #check this could not work
        nfunc=1+(nfunc[:]-1)*(rw[:]**2)

        return nfunc
    
        
    def rectfr(self,tfsupp,TFR,freq,wopt,method):

        #% - returns the component's amplitude [iamp], phase [iphi] and frequency
        #%   [ifreq] as reconstructed from its extracted time-frequency support
        #%   [tfsupp] in the signal's WFT/WT [TFR] (determines whether TFR is WFT
        #%   or WT based on the spacings between specified frequencies [freq] -
        #%   linear (WFT) or logarithmic (WT)). The optional output [rtfsupp]
        #%   returns the extracted time-frequency support if the input [tfsupp]
        #%   specifies 1xL frequency profile instead of the full time-frequency
        #%   support (see below); otherwise returns input [rtfsupp]=[tfsupp].
        #%
        #% INPUT:
        #% tfsupp: 3xL matrix  (or 1xL vector of the desired frequency profile)
        #%        - extracted time-frequency support of the component, containing
        #%          frequencies of the TFR amplitude peaks (ridge points) in the
        #%          first row, support lower bounds (referred as \omega_-(t)/2/pi
        #%          in [1]) - in the second row, and the upper bounds (referred as
        #%          \omega_+(t)/2/pi in [1]) - in the third row. Alternatively, one
        #%          can specify [tfsupp] as 1xL vector of the desired frequency
        #%          profile, in which case the program will automatically select
        #%          time-frequency support around it and the corresponding peaks.
        #% TFR: NFxL matrix (rows correspond to frequencies, columns - to time)
        #%        - time-frequency representation (WFT or WT), to which [tfsupp]
        #%          correspond
        #% freq: NFx1 vector
        #%        - the frequencies corresponding to the rows of [TFR]
        #% wopt: structure returned by function wft.m or wt.m
        #%        - parameters of the window/wavelet and the simulation, returned as
        #%          a third output by functions wft, wt; [wopt] contains all the
        #%          needed information, i.e. name of the TFR, sampling frequency,
        #%          window/wavelet characteristics etc.
        #% method:'direct'(default)|'ridge'|'both'
        #%        - the reconstruction method to use for estimating the component's
        #%          parameters [iamp], [iphi], [ifreq] (see [1]); if set to 'both',
        #%          all parameters are returned as 2xL matrices with direct and
        #%          ridge estimates corresponding to 1st and 2nd rows, respectively.
        #%
        #% NOTE: in the case of direct reconstruction, if the window/wavelet does
        #% not allow direct estimation of frequency, i.e. [wopt.wp.omg=Inf] for
        #% the WFT or [wopt.wp.D=Inf] for the WT (as for the Morlet wavelet),
        #% corresponding to infinite \bar{\omega}_g or D_\psi (in the notation of
        #% [1]), then the frequency is reconstructed by hybrid method (see [1]).
        #%

        [NF,L]=np.shape(TFR)
        freq=freq[:] 
        fs=wopt.fs
        wp=wopt.wp #note wopt class, wp class as well
        method=1;

        #If called from Python, the 'fwt' field is a string.
        #check this in another way
        try:
            wp.fwt = lambda : wp.fwt
        except:
            print('ERROR')
        end

        if method.lower=='direct':
            method=1
        elif method=='ridge':
            method=2
        else:
            method==0

        #% If called from Python, reconstruct complex array.
        #% (MATLAB doesn't allow complex arrays to be passed.)
        # This should not be necessary 
        #if nargin > 5 && ~isempty(varargin{2})
        #    TFR = complex(TFR, varargin{2});
        #end

        idt=np.arange(1,L+1) #check what is the use of idt
        if isinstance(tfsupp,dict):
           idt=tfsupp['2']
           tfsupp=tfsupp['1']

        #%define component parameters and find time-limits
        NR=2
        if method==1 or method==2:
           NR=1
        NC=len(idt)
        mm=np.ones((NR,NC))*math.nan
        ifreq=mm
        iamp=mm
        iphi=mm
        asig=mm
        tn1=np.argwhere( not np.isnan(tfsupp[0,:]))[0] 
        tn2=np.argwhere(not isnan(tfsupp[0,:]))[-1]

        #%Determine the frequency resolution and transform [tfsupp] to indices
        if freq[0]<=0 or np.std(np.diff(freq,1,0))<np.std(freq[1:]/freq[0:-1]):
            fres='linear'
            fstep=np.mean(np.diff(freq,1,0))
            tfsupp=1+np.floor((1/2)+(tfsupp-freq[0])/fstep)
        else:
            fres='log'
            fstep=np.mean(np.diff(np.log(freq),0,1));
            tfsupp=1+floor((1/2)+(np.log(tfsupp)-np.log(freq[0]))/fstep)
       
            #cut-off
        tfsupp[tfsupp<1]=1
        tfsupp[tfsupp>NF]=NF;

        #Define variables to not use cells or structures
        C=wp.C
        ompeak=wp.ompeak
        fwtmax=wp.fwtmax
        if hasattr(wp,'omg'):
           omg=wp.omg
        else:
           D=wp.D
        if not isinstance(wp.fwt, dict):
           fwt=wp.fwt
           nflag=0;
        else:
            nflag=1
            Lf=len(wp.fwt['2'])
            if fres.lower=='linear':
               fxi=wp.fwt['2'] 
               fwt=wp.fwt['1']
            else:
                fxi=np.linspace(np.amin(wp.fwt['2']),np.amax(wp.fwt['2']),Lf).T
                fwt=np.interp(fxi,wp.fwt['2'],wp.fwt['1']) #this is linear interpolation, not using splines #TODO
            
            wstep=np.mean(np.diff(fxi,1,0))
        

        #If only the frequency profile is specified, extract the full time-frequency support
        if np.amin(np.shape(tfsupp))==1:
            eind=tfsupp[:].T
            tfsupp=np.zeros((3,L))*math.nan
            for tn in range(tn1,tn2+1):
                cind=eind[tn] 
                xn=idt[tn] 
                cs=np.abs(TFR[:,xn])
        
                #Ridge point
                cpeak=cind
                if cind>1 and cind<NF:
                    if cs[cind+1]==cs[cind-1]: #from here 14/5
                        cpeak1=cind-1+np.ravel_multi_index(np.nonzero((cs[cind:-1]>=cs[cind-1:-2] and cs[cind:-1]>cs[cind+1:])), np.shape(cs))[0]
                        cpeak1=np.amin([cpeak1,NF])
                        cpeak2=cind+1-np.ravel_multi_index(np.nonzero(cs[cind:0:-1]>=cs[cind+1:1:-1] and cs[cind:0:-1]>cs[cind-1:0:-1],np.shape(cs)))[0] 
                        cpeak2=np.amax([cpeak2,1]);
                        if cs[cpeak1]>0 and cs[cpeak2]>0:
                            if cpeak1-cind==cind-cpeak2:
                                if cs[cpeak1]>cs[cpeak2]: 
                                    cpeak=cpeak1;
                                else:
                                   cpeak=cpeak2
                            elif cpeak1-cind<cind-cpeak2:
                               cpeak=cpeak1
                            elif cpeak1-cind>cind-cpeak2:
                                cpeak=cpeak2
                            
                        elif cs[cpeak1]==0:
                           cpeak=cpeak2
                        elif cs[cpeak2]==0:
                           cpeak=cpeak1
                        
                    elif cs[cind+1]>cs[cind-1]:
                        cpeak=cind-1+np.ravel_multi_index(np.nonzero(cs[cind:-1]>=cs[cind-1:-2] and cs[cind:-1]>cs[cind+1:],np.shape(cs)))[0] 
                        cpeak=np.amin([cpeak,NF])
                    elif cs[cind+1]<cs[cind-1]:
                        cpeak=cind+1-np.ravel_multi_index(np.nonzero(cs[cind:0:-1]>cs[cind-1::-1] and cs[cind:0:-1]>=cs[cind+1:1:-1],1,np.shape(cs)))[0]
                        cpeak=np.amax([cpeak,1])
                    
                elif cind==1:
                    if cs[1]<cs[0]:
                       cpeak=cind
                    else:
                        cpeak=1+np.ravel_multi_index(np.nonzero(cs[cind+1:-1]>=cs[cind:-2] and cs[cind+1:-1]>cs[cind+2:],np.shape(cs)))[0] 
                        cpeak=np.amin([cpeak,NF])
                    
                elif cind==NF:
                    if cs[NF-1]<cs[NF]:
                        cpeak=cind
                    else:
                        cpeak=NF-np.ravel_multi_index(np.nonzero(cs[cind-1:0:-1]>cs[cind-2::-1] and cs[cind-1:0:-1]>=cs[cind:1:-1],np.shape(cs)))[0] 
                        cpeak=np.max([cpeak,1])
                    
                
                tfsupp[0,tn]=cpeak
        
                #Boundaries of time-frequency support
                iup=[]
                idown=[]
                if cpeak<NF-1:
                   iup=cpeak+np.ravel_multi_index(np.nonzero(cs[cpeak+1:-1]<=cs[cpeak:-2] and cs[cpeak+1:-1]<cs[cpeak+2:],np.shape(cs)))[0]
                if cpeak>2:
                    idown=cpeak-np.ravel_multi_index(np.nonzero(cs[cpeak-1:0:-1]<=cs[cpeak:1:-1] and cs[cpeak-1:0:-1]<cs[cpeak-2::-1],np.shape(cs)))[0]
                iup=np.amin([iup,NF])
                idown=np.amax([idown,1])
                tfsupp[1,tn]=idown
                tfsupp[2,tn]=iup
            
        
        #%If only the boundaries of the frequency profile are specified, extract the peaks
        if np.amin(np.shape(tfsupp))==2:
            pind=np.zeros((1,L))*math.nan
            for tn in range(tn1,tn2+1):
                ii=np.arange(tfsupp[0,tn],tfsupp[1,tn]) 
                xn=idt[tn] 
                cs=np.abs(TFR[ii,xn])
                mid=np.np.unravel_index(np.argmax(cs))[1]
                pind[tn]=ii[mid]
            
            tfsupp=[[pind],[tfsupp]]
        
        #%Return extracted time-frequency support if requested
        #if nargout>3, varargout{1}=tfsupp*NaN; varargout{1}(:,tn1:tn2)=freq(tfsupp(:,tn1:tn2)); end
        rtfsupp[:,tn1:tn2+1]=freq[tfsupp[:,tn1:tn2+1]]
        #%==================================WFT=====================================
        if fres.lower=='linear':
    
            #Direct reconstruction-------------------------------------------------
            if method==0 or method==1:
                for tn in range(tn1,tn2+1):
                    ii=np.arange(tfsupp[1,tn],tfsupp[2,tn])
                    xn=idt[tn] 
                    cs=TFR[ii,xn]
                    if np.isfinite(omg):
                        if xn>idt[tn1] and xn<idt[tn2]:
                            cw=np.angle(TFR[ii,xn+1])-np.angle(TFR[ii,xn-1]) #np.angle -> phase angle
                            cw[cw<0]=cw[cw<0]+2*np.pi
                            cw=cw*fs/2; 
                        elif xn==1: 
                            cw=np.angle(TFR[ii,xn+1])-np.angle(cs)
                            cw[cw<0]=cw[cw<0]+2*np.pi
                            cw=cw*fs
                        else: 
                            cw=np.angle(cs)-np.angle(TFR[ii,xn-1]); 
                            cw[cw<0]=cw[cw<0]+2*np.pi
                            cw=cw*fs 
                        cw=cw/(2*np.pi)
                    
            
                    if cs[np.isnan(cs)]==[] or not np.isfinite(cs):
                        casig=(1/C)*np.sum(cs*2*np.pi*fstep) 
                        asig[0,tn]=casig
                        if np.isfinite(omg):
                            ifreq[0,tn]=-omg+(1/C)*np.sum(freq[ii]*cs*(2*np.pi*fstep))/casig
                        else:
                            ifreq[0,tn]=(1/C)*np.sum(cw*cs*(2*pi*fstep))/casig

                ifreq[0,:]=np.real(ifreq[0,:]) #real part
            
    
            #%Ridge reconstruction--------------------------------------------------
            if method==0 or method==2:
                rm=2
                if method==2:
                    rm=1
                for tn in range(tn1,tn2+1):
                    ipeak=tfsupp[1,tn]
                    xn=idt[tn]
                    if ipeak>1 and ipeak<NF:
                       cs=TFR[[[ipeak],[ipeak-1],[ipeak+1]],xn]
                    else:
                       cs=TFR[ipeak,xn]
                
                    if isfinite(cs[0]):
                        ifreq[rm,tn]=freq[ipeak]-ompeak/2/np.pi
                        if ipeak>1 and ipeak<NF: #%quadratic interpolation
                            a1=np.abs(cs[1]) 
                            a2=abs(cs[0])
                            a3=abs(cs[2])
                            p=(1/2)*(a1-a3)/(a1-2*a2+a3)
                            if np.abs(p)<=1:
                                ifreq[rm,tn]=ifreq[rm,tn]+p*fstep
                        
                        ximax=2*np.pi*(freq[ipeak]-ifreq[rm,tn])
                        if nflag==0: #%if window FT is known in analytic form
                            cmax=fwt[ximax]
                            if np.isnan(cmax):
                               cmax=fwt[ximax+10**(-14)] 
                            if np.isnan(cmax):
                               cmax=fwtmax
                        else: #%if window FT is numerically estimated
                            cid1=1+np.floor((ximax-fxi[0])/wstep) 
                            cid2=cid1+1;
                            cid1=np.amin([np.amax([cid1,1]),Lf]) 
                            cid2=np.amin([np.amax([cid2,1]),Lf])
                            if cid1==cid2:
                                cmax=fwt[cid1]
                            else:
                                cmax=fwt[cid1]+(fwt[cid2]-fwt[cid1])*(ximax-fxi[cid1])/(fxi[cid2]-fxi[cid1])
                            
                        
                        casig=2*cs[0]/cmax
                        asig[rm,tn]=casig
  
        #%===================================WT=====================================
        if fres.lower=='log':
    
            #%Direct reconstruction-------------------------------------------------
            if method==0 or method==1:
                for tn in range(tn1,tn2+1):
                    ii=np.arange(tfsupp[1,tn],tfsupp[2,tn])
                    xn=idt[tn] 
                    cs=TFR[ii,xn]
                    if np.isfinite(D):
                        if xn>idt[tn1] and xn<idt[tn2]:
                           cw=np.angle(TFR[ii,xn+1])-np.angle(TFR[ii,xn-1])
                           cw[cw<0]=cw[cw<0]+2*np.pi 
                           cw=cw*fs/2
                        elif xn==1:
                           cw=np.angle(TFR[ii,xn+1])-np.angle(cs)
                           cw[cw<0]=cw[cw<0]+2*np.pi
                           cw=cw*fs
                        else:
                           cw=np.angle(cs)-np.angle(TFR[ii,xn-1])
                           cw[cw<0]=cw[cw<0]+2*np.pi
                           cw=cw*fs
                        cw=cw/(2*np.pi)
                    
            
                    if isempty(cs[np.isnan(cs)] or not np.isfinite(cs)):
                        casig=(1/C)*np.sum(cs*fstep)
                        asig[0,tn]=casig
                        if np.isfinite(D):
                            ifreq[0,tn]=(1/D)*np.sum(freq[ii]*cs*fstep)/casig
                        else:
                            ifreq[0,tn]=(1/C)*np.sum(cw*cs*fstep)/casig
                        
                ifreq[0,:]=np.real(ifreq[0,:])
            
    
            #%Ridge reconstruction--------------------------------------------------
            if method==0 or method==2:
                rm=2
                if method==2:
                   rm=1
                for tn in range(tn1,tn2+1):
                    ipeak=tfsupp[0,tn] 
                    xn=idt[tn]
                    if ipeak>1 or ipeak<NF:
                       cs=TFR([[ipeak],[ipeak-1],[ipeak+1]],xn)
                    else:
                       cs=TFR[ipeak,xn]
            
                    if isfinite(cs[0]):
                        ifreq[rm,tn]=freq[ipeak]
                        if ipeak>1 and ipeak<NF: #%quadratic interpolation
                            a1=np.abs(cs[1])
                            a2=np.abs(cs[0])
                            a3=np.abs(cs[2])
                            p=(1/2)*(a1-a3)/(a1-2*a2+a3)
                            if np.abs(p)<=1:
                                np.ifreq[rm,tn]=np.exp(np.log(ifreq[rm,tn])+p*fstep)
                        ximax=ompeak*ifreq[rm,tn]/freq[ipeak]
                        if nflag==0: #%if wavelet FT is known in analytic form
                            cmax=fwt[ximax] 
                            if np.isnan(cmax):
                                cmax=fwt[ximax*(1+10**(-14))]
                            if np.isnan(cmax):
                               cmax=fwtmax
                        else: #%if wavelet FT is numerically estimated
                            cid1=1+np.floor((ximax-fxi[0])/wstep)
                            cid2=cid1+1
                            cid1=np.amin([np.amax([cid1,1]),Lf])
                            cid2=np.amin([np.amax([cid2,1]),Lf])
                            if cid1==cid2:
                                cmax=fwt[cid1]
                            else:
                                cmax=fwt[cid1]+(fwt[cid2]-fwt[cid1])*(ximax-fxi[cid1])/(fxi[cid2]-fxi[cid1])
                        casig=2*cs[0]/cmax
                        asig[rm,tn]=casig
        #%Estimate amplitude and phase (faster to do all at once)
        iamp=np.abs(asig)
        iphi=np.angle(asig);
        #%Unwrap phases at the end
        for sn in range(1,np.shape(iphi)[0]+1):
            iphi[sn,:]=np.unwrap(iphi[sn,:]) #should be equivalent to MATLAB inwrap

        return iamp,iphi,ifreq,rtfsupp
      

                


    


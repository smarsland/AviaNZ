
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
        tn1=np.argwhere(np.isnan(TFR[-1,:]))[0] #find first index where TFR is NAN of last line
        tn2=np.argwhere(np.isnan(TFR[-1,:]))[-1] #find last index where TFR is NAN of last line
        sflag=0;
        

        if (type(method) is char and method.lower!='max') or length(method)==1: #if not frequency-based or maximum-based extraction
            sflag=1
            #----------------------------------------------------------------------
            #Construct matrices of ridge indices, frequencies and amplitudes:
            #[Ip],[Fp],[Qp], respectively; [Np] - number of peaks at each time.
            if ~isempty(Skel):
                Np=Skel{1}; Ip=Skel{2}; Fp=Skel{3}; Qp=Skel{4}; Mp=max(Np);
            else
                if ~strcmpi(DispMode,'off') && ~strcmpi(DispMode,'notify')
                    fprintf('Locating the amplitude peaks in TFR... ');
                end
                TFR=vertcat(zeros(1,L),TFR,zeros(1,L)); %pad TFR with zeros
                idft=1+find(TFR(2:end-1)>=TFR(1:end-2) & TFR(2:end-1)>TFR(3:end)); %find linear indices of the peaks
                [idf,idt]=ind2sub(size(TFR),idft); idf=idf-1; %find frequency and time indices of the peaks
                idb=find(idf==1 | idf==NF); idft(idb)=[]; idf(idb)=[]; idt(idb)=[]; %remove the border peaks
                dind=[0;find(diff(idt(:))>0);length(idt)]; Mp=max([max(diff(dind)),2]);
                Np=zeros(1,L); idn=zeros(length(idt),1);
                for dn=1:length(dind)-1,
                    ii=dind(dn)+1:dind(dn+1); idn(ii)=1:length(ii);
                    Np(idt(ii(1)))=length(ii);
                end
                idnt=sub2ind([Mp,L],idn(:),idt(:));
                %Quadratic interpolation to better locate the peaks
                a1=TFR(idft-1); a2=TFR(idft); a3=TFR(idft+1);
                dp=(1/2)*(a1-a3)./(a1-2*a2+a3);
                %Assign all
                Ip=ones(Mp,L)*NaN; Fp=ones(Mp,L)*NaN; Qp=ones(Mp,L)*NaN;
                Ip(idnt)=idf+dp; Qp(idnt)=a2-(1/4)*(a1-a3).*dp;
                if fres==1, Fp(idnt)=freq(idf)+dp(:)*fstep;
                else Fp(idnt)=freq(idf).*exp(dp(:)*fstep); end
                %Correct "bad" places, if present
                idb=find(isnan(dp) | abs(dp)>1 | idf==1 | idf==NF);
                if ~isempty(idb)
                    Ip(idnt(idb))=idf(idb);
                    Fp(idnt(idb))=freq(idf(idb));
                    Qp(idnt(idb))=a2(idb);
                end
                %Remove zeros and clear the indices
                TFR=TFR(2:end-1,:); clear idft idf idt idn dind idnt a1 a2 a3 dp;
                %Display
                if ~strcmpi(DispMode,'off')
                    if ~strcmpi(DispMode,'notify')
                        fprintf('(number of ridges: %d+-%d, from %d to %d)\n',round(mean(Np(tn1:tn2))),round(std(Np(tn1:tn2))),min(Np),max(Np));
                    end
                    idb=find(Np(tn1:tn2)==0); NB=length(idb);
                    if NB>0, fprintf(2,sprintf('Warning: At %d times there are no peaks (using border points instead).\n',NB)); end
                end
                %If there are no peaks, assign border points
                idb=find(Np(tn1:tn2)==0); idb=tn1-1+idb; NB=length(idb);
                if NB>0, G4=abs(TFR([1;2;NF-1;NF],idb)); end
                for bn=1:NB
                    tn=idb(bn); cn=1; cg=G4(:,bn);
                    if cg(1)>cg(2) || cg(4)>cg(3)
                        if cg(1)>cg(2), Ip(cn,tn)=1; Qp(cn,tn)=cg(1); Fp(cn,tn)=freq(1); cn=cn+1; end
                        if cg(4)>cg(3), Ip(cn,tn)=NF; Qp(cn,tn)=cg(4); Fp(cn,tn)=freq(NF); cn=cn+1; end
                    else
                        Ip(1:2,tn)=[1;NF]; Qp(1:2,tn)=[cg(1);cg(4)]; Fp(1:2,tn)=[freq(1);freq(NF)]; cn=cn+2;
                    
                    Np(tn)=cn-1;
                
                clear idb NB G4;
            if nargout>2, varargout{2}={Np,Ip,Fp,Qp}; end
            if strcmpi(NormMode,'on'), nfunc=tfrnormalize(abs(TFR(:,tn1:tn2)),freq); end
            ci=Ip; ci(isnan(ci))=NF+2; cm=ci-floor(ci); ci=floor(ci); nfunc=[nfunc(1);nfunc(:);nfunc(end);NaN;NaN];
            Rp=(1-cm).*nfunc(ci+1)+cm.*nfunc(ci+2); Wp=AmpFunc(Qp.*Rp); nfunc=nfunc(2:end-3); %apply the functional to amplitude peaks
    
        elseif ~ischar(method) && length(method)>1 %frequency-based extraction
            if length(method)~=L
                error('The specified frequency profile ("Method" property) should be of the same length as signal.');
            
    
            efreq=method; submethod=1; if max(abs(imag(efreq)))>0, submethod=2; efreq=imag(efreq); end
            if ~strcmpi(DispMode,'off') && ~strcmpi(DispMode,'notify')
                if submethod==1, fprintf('Extracting the ridge curve lying in the same TFR supports as the specified frequency profile.\n');
                else fprintf('Extracting the ridge curve lying nearest to the specified frequency profile.\n'); end
            
    
            tn1=max([tn1,find(~isnan(efreq),1,'first')]); tn2=min([tn2,find(~isnan(efreq),1,'last')]);
            if fres==1, eind=1+floor(0.5+(efreq-freq(1))/fstep);
            else eind=1+floor(0.5+log(efreq/freq(1))/fstep); end
            eind(eind<1)=1; eind(eind>NF)=NF;
    
            %Extract the indices of the peaks
            for tn=tn1:tn2
                cind=eind(tn); cs=abs(TFR(:,tn));
        
                %Ridge point
                cpeak=cind;
                if cind>1 && cind<NF
                    if cs(cind+1)==cs(cind-1) || submethod==2
                        cpeak1=cind-1+find(cs(cind:end-1)>=cs(cind-1:end-2) & cs(cind:end-1)>cs(cind+1:end),1,'first'); cpeak1=min([cpeak1,NF]);
                        cpeak2=cind+1-find(cs(cind:-1:2)>=cs(cind+1:-1:3) & cs(cind:-1:2)>cs(cind-1:-1:1),1,'first'); cpeak2=max([cpeak2,1]);
                        if cs(cpeak1)>0 && cs(cpeak2)>0
                            if cpeak1-cind==cind-cpeak2
                                if cs(cpeak1)>cs(cpeak2), cpeak=cpeak1;
                                else cpeak=cpeak2; end
                            elseif cpeak1-cind<cind-cpeak2, cpeak=cpeak1;
                            elseif cpeak1-cind>cind-cpeak2, cpeak=cpeak2;
                            end
                        elseif cs(cpeak1)==0, cpeak=cpeak2;
                        elseif cs(cpeak2)==0, cpeak=cpeak1;
                        end
                    elseif cs(cind+1)>cs(cind-1)
                        cpeak=cind-1+find(cs(cind:end-1)>=cs(cind-1:end-2) & cs(cind:end-1)>cs(cind+1:end),1,'first'); cpeak=min([cpeak,NF]);
                    elseif cs(cind+1)<cs(cind-1)
                        cpeak=cind+1-find(cs(cind:-1:2)>cs(cind-1:-1:1) & cs(cind:-1:2)>=cs(cind+1:-1:3),1,'first'); cpeak=max([cpeak,1]);
                    end
                elseif cind==1
                    if cs(2)<cs(1), cpeak=cind;
                    else
                        cpeak=1+find(cs(cind+1:end-1)>=cs(cind:end-2) & cs(cind+1:end-1)>cs(cind+2:end),1,'first'); cpeak=min([cpeak,NF]);
                    end
                elseif cind==NF
                    if cs(NF-1)<cs(NF), cpeak=cind;
                    else
                        cpeak=NF-find(cs(cind-1:-1:2)>cs(cind-2:-1:1) & cs(cind-1:-1:2)>=cs(cind:-1:3),1,'first'); cpeak=max([cpeak,1]);
                    end
                end
                tfsupp(1,tn)=cpeak;
        
                %Boundaries of time-frequency support
                iup=[]; idown=[];
                if cpeak<NF-1, iup=cpeak+find(cs(cpeak+1:end-1)<=cs(cpeak:end-2) & cs(cpeak+1:end-1)<cs(cpeak+2:end),1,'first'); end
                if cpeak>2, idown=cpeak-find(cs(cpeak-1:-1:2)<=cs(cpeak:-1:3) & cs(cpeak-1:-1:2)<cs(cpeak-2:-1:1),1,'first'); end
                iup=min([iup,NF]); idown=max([idown,1]);
                tfsupp(2,tn)=idown; tfsupp(3,tn)=iup;
            end
    
            %Transform to frequencies
            pind=tfsupp(1,:); tfsupp(:,tn1:tn2)=freq(tfsupp(:,tn1:tn2));
            pamp(tn1:tn2)=abs(TFR(sub2ind(size(TFR),pind(tn1:tn2),tn1:tn2)));
    
            %Optional output arguments
            if nargout>1
                ec.efreq=efreq; ec.eind=eind;
                ec.pfreq=tfsupp(1,:); ec.pind=pind; ec.pamp=pamp; ec.idr=idr;
                varargout{1}=ec;
            end
            if nargout>2, varargout{2}=[]; end
    
            %Plotting (if needed)
            if ~isempty(strfind(DispMode,'plot'))
                scrsz=get(0,'ScreenSize'); figure('Position',[scrsz(3)/4,scrsz(4)/8,2*scrsz(3)/3,2*scrsz(4)/3]);
                ax=axes('Position',[0.1,0.1,0.8,0.8],'FontSize',16); hold all;
                title(ax(1),'Ridge curve \omega_p(t)/2\pi'); ylabel(ax(1),'Frequency (Hz)'); xlabel(ax(1),'Time (s)');
                plot(ax(1),(0:L-1)/fs,efreq,'--','Color',[0.5,0.5,0.5],'LineWidth',2,'DisplayName','Specified frequency profile');
                plot(ax(1),(0:L-1)/fs,tfsupp(1,:),'-k','LineWidth',2,'DisplayName','Extracted frequency profile');
                legend(ax(1),'show'); if fres==2, set(ax(1),'YScale','log'); end
            end
            if ~isempty(strfind(PlotMode,'on')), plotfinal(tfsupp,TFR,freq,fs,DispMode,PlotMode); end
            if nargout>2, varargout{2}=Skel; end
    
            return;
    
        return tfsupp,ecinfo,Skel




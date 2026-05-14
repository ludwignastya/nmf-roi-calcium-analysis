from numpy import *
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from scipy import signal

# -------------------------------
# Main class for ROI trace analysis
# -------------------------------

class Traces:
    def __init__(self, movie, mask, t=100):
        
        """
        Initialize the Traces object.

        movie : 3D array (time × height × width)
        mask  : 2D array (binary mask defining ROIs)
        t     : frame duration in ms (default = 100 → 10 Hz sampling)
        """

        self.frame_rate=t
        # Find coordinates of mask (non-zero pixels → ROIs)
        barrels=nonzero(mask)
        barrels=[(barrels[0][i], barrels[1][i]) for i in range(len(barrels[0]))]
        self.traces=[]
        self.raw_traces=[]
        self.roi=[]
        h=movie.shape[1]
        w=movie.shape[2]
        # Extract time series for each pixel inside mask
        for y in range(h):
            for x in range(w):
                if (y,x) in barrels:
                    self.traces.append(movie[:,y,x])
                    self.roi.append(y*w+x)
        self.traces=np.array(self.traces)
        self.nframes = movie.shape[0]
        self.nroi=self.traces.shape[0]

    # -------------------------------
    # Filtering methods
    # -------------------------------

    # High-pass filter 
    def HPfilter(self,cutoff):
        order=5
        fps=1000/self.frame_rate
        b, a = signal.butter(order, cutoff, fs=fps, btype='high', analog=False)
        corr=np.empty([self.nroi,self.nframes])
        for roi in range(self.nroi):
            trace=self.traces[roi]
            filtered_trace = signal.filtfilt(b, a, trace)+trace[0]
            if filtered_trace.mean()==0:
                print(f"roi {roi} mean is 0; could not normalize")
            else:
                corr[roi]=(filtered_trace-filtered_trace.mean())/abs(filtered_trace.mean())
        self.traces=corr

    # Low-pass filter 
    def LPfilter(self,cutoff):
        order=5
        fps=1000/self.frame_rate
        b, a = signal.butter(order, cutoff, fs=fps, btype='low', analog=False)
        corr=np.empty([self.nroi,self.nframes])
        for roi in range(self.nroi):
            trace=self.traces[roi]
            filtered_trace = signal.filtfilt(b, a, trace)
            if filtered_trace.mean()==0:
                print(f"roi {roi} mean is 0; could not normalize")
            else:
                corr[roi]=(filtered_trace-filtered_trace.mean())/abs(filtered_trace.mean())
        self.traces=corr

    # Band-pass filter 
    def BPfilter(self, order, cutoff):
        fps=1000/self.frame_rate
        b, a = signal.butter(order, cutoff, fs=fps, btype='band', analog=False)
        corr=np.empty([self.nroi,self.nframes])
        for roi in range(self.nroi):
            trace=self.traces[roi]
            filtered_trace = signal.filtfilt(b, a, trace)+trace[0]
            if filtered_trace.mean()==0:
                print(f"roi {roi} mean is 0; could not normalize")
            else:
                corr[roi]=(filtered_trace-filtered_trace.mean())/abs(filtered_trace.mean())
        self.traces=corr

    # Linear detrending
    def detrend(self,factor=1):
        def func1(a,b,x):
            return x*a#+b
        def func2(a,b,c,x):
            return pow(x,2)*a+x*b#+c
        def func3(a,b,c,d,x):
            return pow(x,3)*a+pow(x,2)*b+x*c#+d    
        def func4(a,b,c,d,e,x):
            return pow(x,4)*a+pow(x,3)*b+pow(x,2)*c+x*d#+e        
        if factor in [0,1,2,3,4]:
            time=np.array(range(0,self.nframes))
            corr=np.empty([self.nroi,self.nframes])
            for roi in range(self.nroi):
                trace=self.traces[roi,:]
                if factor==0:
                    for t in range(self.nframes):
                        corr[roi,t]=trace[t]
                if factor==1:
                    coef=np.polyfit(time,trace,1)                 
                    for t in range(self.nframes):
                        corr[roi,t]=trace[t]-func1(*coef, t)
                elif factor==2:
                    coef=np.polyfit(time,trace,2)
                    for t in range(self.nframes):
                        corr[roi,t]=trace[t]-func2(*coef, t)
                elif factor==3:
                    coef=np.polyfit(time,trace,3)
                    for t in range(self.nframes):
                        corr[roi,t]=trace[t]-func3(*coef, t)
                else:
                    coef=np.polyfit(time,trace,4)
                    for t in range(self.nframes):
                        corr[roi,t]=trace[t]-func4(*coef, t)

            corr_mean=corr.mean(axis=1)
            for roi in range(self.nroi):
                if corr_mean[roi]!=0:
                    corr[roi]=(corr[roi]-corr_mean[roi])/abs(corr_mean[roi])
            self.traces=corr
        else:
            print("for detrending enter factor 0 (no detrending), 1 (linear regression),2(quadratic regression), 3, or 4")    
    
    def remove_low(self, degree):
        mn=self.traces.mean(axis=1)
        sd=self.traces.std(axis=1)
        for roi in range(self.nroi):
            signal=self.traces[roi]
            thresh=mn[roi]-degree*sd[roi]
            signal=np.where(signal<thresh,np.nan,signal)
            mni=nanmin(signal)
            signal=np.where(signal!=signal,mni,signal)
            self.traces[roi]=signal

    
    # -------------------------------
    # Peak detection
    # -------------------------------

    def get_peaks_maxsegments(self, thsh=80, strict=3, dF_thresh=0.05):
        # Initialize storage containers 
        self.precentile_threshold=thsh      
        self.peaks={} # Dictionary to store detected peaks for each ROI; Format: {roi_index: [(start, max, end), ...]}
        self.amp={} # Dictionary to store peak amplitudes for each ROI; Format: {roi_index: [amplitude1, amplitude2, ...]}
        
        # --- Auxiliary function: find contiguous segments above/below threshold ---
        def find_segments(signal, thresh, sign):
            temp=[]
            itemp=[]
            segments=[]
            flag=False
            # Define condition depending on sign
            if sign=='>':
                crit=signal>thresh
            elif sign=='<':
                crit=signal<thresh
            else:
                raise ValueError('Unknown sign')

            # Iterate through signal
            for i in range(len(signal)):
                if crit[i]:
                    # Inside segment -> collect values and indices
                    temp.append(signal[i])
                    itemp.append(i)
                    flag=True
                else:
                    # End of segment -> store i
                    if flag:
                        segments.append(np.array([temp,itemp]))
                        temp=[]
                        itemp=[]
                        flag=False
            # Keep only segments with more than one sample
            segments=[s for s in segments if len(s[0])>1]
            return segments    

        # --- Step 1: Estimate baseline using percentile ---        
        pers=np.percentile(self.traces, self.precentile_threshold, axis=1) # Compute percentile threshold per ROI (row-wise)
        pers=np.reshape(pers,(len(pers),1)) # # Reshape for broadcasting (Nroi × 1)
        filtered=np.where(self.traces<pers,self.traces,np.nan) ## Keep only values below percentile → assumed baseline
        mn=nanmean(filtered, axis=1) # baseline mean per ROI
        sd=nanstd(filtered,axis=1) # baseline std per ROI
        
        # --- Step 2: Process each ROI separately ---
        for roi in range(len(self.traces)):
            # Find segments above threshold (mean + strict × SD
            segmax=find_segments(self.traces[roi], mn[roi]+strict*sd[roi], '>')

            # --- Step 3: Find peak maxima indices ---
            mxi=[] #indices of maximums      
            for s in segmax:
                mx=max(s[0])
                for i in range(len(s[1])):
                    if s[0][i]==mx:
                        mxi.append(int(s[1][i]))    

            # --- Step 4: Find peak start and end indices ---
            mni=[] #indices of minimums (beginings of peaks)
            mne=[] #indices of ends of peaks

            for mi in mxi:
                # --- Backtrack from max to find start --                        
                x=mi
                while x>0 and self.traces[roi][x]>mn[roi]+sd[roi]:
                    x-=1
                    # If another peak encountered → adjust to local minimum
                    if x in mxi:
                        x=x+np.argmin(self.traces[roi][x:mi])
                        break
                mni.append(x)

                # --- Forward tracking to find end ---
                x=mi
                while x<len(self.traces[roi]) and self.traces[roi][x]>mn[roi]+sd[roi]:
                    x+=1
                    if x in mxi:
                        x=x-len(self.traces[roi][mi:x])+np.argmin(self.traces[roi][mi:x])
                        break
                mne.append(x) 

            # --- Step 5: Filter peaks by amplitude -
            peaks=[]
            amp=[]
            for i in range(len(mxi)):
                # Keep only peaks above amplitude threshold
                if self.traces[roi][mxi[i]]>dF_thresh:
                    peaks.append((mni[i],mxi[i],mne[i]))
                    amp.append(self.traces[roi][mxi[i]])
            
            # Store results for this ROI
            self.peaks[roi]=peaks
            self.amp[roi]=amp
         
    def find_active_roi(self):        
        # Initialize list of active ROIs
        # Active ROI = ROI that contains at least one detected peak
        self.active_roi=[]
        for roi in self.peaks:
            # If the ROI has at least one peak (non-empty list)
            if self.peaks[roi]:
                self.active_roi.append(roi) # mark ROI as active

 
    # -------------------------------
    # Activity analysis
    # -------------------------------

    def find_raster(self,rois='all', prn=True):
        # If no specific ROI subset is provided, use all ROI
        if rois=='all':
            rois=list(range(self.nroi))
            
        # Initialize raster matrix:
        # shape = (time × number_of_ROIs)
        # values = 1 if a peak maximum occurs at that time for that ROI
        self.raster=np.zeros((self.nframes, len(self.roi)))
        # Fill raster matrix with peak maxima
        for roi in rois:
            mx=[peak[1] for peak in self.peaks[roi]] ## Extract peak maximum indices for this ROI
            self.raster[mx,roi]=1  # Mark those time points as active (1)

        # If plotting is enabled
        if prn:
            # Create figure for raster plot
            fig, ax1 = plt.subplots(1,1,figsize=[10,5])
            # Plot raster for each ROI
            for roi in rois:
                # Convert frame indices to time (seconds) where peaks occur
                ind=[i*(self.frame_rate/1000) for i,e in enumerate(self.raster[:,roi]>0) if e]
                # Plot each peak as a dot (time vs ROI index)
                ax1.plot(ind,[roi]*len(ind),'.k')
                # Label axes and title
                ax1.set_xlabel('time (sec)', fontsize=16)
                ax1.set_ylabel('ROI #', fontsize=16)
                ax1.set_title('raster plot of ROIs inside the masked region', fontsize=16)
                 # Set font size of tick labels
                for label in (ax1.get_xticklabels() + ax1.get_yticklabels()): label.set_fontsize(12)

    def export_intdata(self):
        # Convert frame duration from ms to seconds
        t=self.frame_rate/1000 
        # Initialize list to store inter-event intervals
        interval=[]
        # Loop over all ROIs
        for roi in self.peaks:
            # Extract peak maxima indices (time points of peaks)
            start=np.array([peak[1] for peak in self.peaks[roi]])
            # Compute differences between consecutive peaks (in frames)
            inter=list(start[1:]-start[:-1])
            # Only include intervals from active ROIs
            if  roi in self.active_roi:
                interval+=[i*t for i in inter] # Convert intervals to seconds and add to global list
        return interval

    def find_active_frames(self):
        # Initialize list to store activity fraction per ROI
        activefr=[]
        # Loop over each ROI (columns of raster matrix)
        for roi in self.raster.transpose():
            # Compute fraction of frames where ROI is active (sum of active frames divided by total frames)
            activefr.append(sum(roi)/self.nframes)
        return activefr




"""
functions_for_NMF_based_Ca_analysis_workflow.py

Utility functions for NMF-based calcium imaging data analysis workflow.

This module provides a collection of functions to:
- Load and preprocess calcium imaging movies (e.g., TIFF stacks)
- Compute ΔF/F (dF/F) signals with masking
- Perform Non-negative Matrix Factorization (NMF)
- Construct and refine spatial component masks
- Split components into spatial blobs
- Compute similarity metrics (e.g., Jaccard index)
- Cluster and merge overlapping components
- Visualize intermediate and final results

The workflow is designed for:
1. Preprocessing calcium imaging data
2. Extracting spatial and temporal components via NMF
3. Post-processing components into biologically meaningful regions

Dependencies
------------
- numpy
- scipy
- scikit-learn
- tifffile
- matplotlib
- pickle
- collections

Author
------
Anastasia Ludwig

Created
-------
12.05.2026

Notes
-----
- Functions are modular and can be used independently or as part of the full workflow.
- Assumes input data is pre-aligned and motion-corrected.
- Masking is required for efficient and accurate decomposition.
"""
# Import 
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd



def compute_dff(movie, mask, percentile=10, eps=1e-6,
                       return_full=False, dtype=np.float32):
    """
    Compute ΔF/F using a spatial mask.

    Parameters
    ----------
    movie : ndarray, shape (T, H, W)
        Raw movie (float32 recommended).
    mask : ndarray, shape (H, W)
        Binary mask (1 = keep pixel).
    percentile : float
        Baseline percentile (e.g., 10).
    eps : float
        Small value to avoid division by zero.
    return_full : bool
        If False: return masked ΔF/F as (T, Npix).
        If True: return full ΔF/F movie (T, H, W).
    dtype : numpy dtype
        Output dtype, default float32.

    Returns
    -------
    dff : ndarray
        ΔF/F, either masked (T, Npix) or full (T, H, W)
    F0 : ndarray
        Baseline fluorescence per masked pixel (Npix,)
    """
    T, H, W = movie.shape
    movie = movie.astype(dtype, copy=False)

    # Flatten spatial dimensions
    movie_flat = movie.reshape(T, -1)          # (T, H*W)
    mask_flat  = mask.reshape(-1).astype(bool)

    # Keep only masked pixels
    movie_masked = movie_flat[:, mask_flat]    # (T, Npix)

    # Compute baseline per pixel (percentile over time)
    F0 = np.percentile(movie_masked, percentile, axis=0)
    F0 = np.maximum(F0, eps)

    # ΔF/F in-place to minimize memory usage
    dff_masked = (movie_masked - F0) / F0
    dff_masked = dff_masked.astype(dtype, copy=False)

    if not return_full:
        return dff_masked, F0

    # Reconstruct full movie if needed
    dff_full = np.zeros((T, H * W), dtype=dtype)
    dff_full[:, mask_flat] = dff_masked
    dff_full = dff_full.reshape(T, H, W)

    return dff_full, F0

def expand_to_full_maps(A_masked, mask):
    """
    A_masked: (Nmask, nIC)
    mask: (H, W) bool
    Returns: maps_full (nIC, H, W)
    """
    H, W = mask.shape
    nIC  = A_masked.shape[1]
    maps_full = np.zeros((nIC, H*W), dtype=A_masked.dtype)
    mflat = mask.reshape(-1).astype(bool)
    maps_full[:, mflat] = A_masked.T
    return maps_full.reshape(nIC, H, W)

def make_nmf_mask(spatial_maps_masked, top_pct=2):
    out=np.asarray(spatial_maps_masked)
    for comp in range(spatial_maps_masked.shape[0]):
        thr_abs = np.percentile(spatial_maps_masked[comp], 100 - float(top_pct))
        roi = spatial_maps_masked[comp] >= thr_abs
        out[comp]=roi
    return out

def expand_to_full_maps(A_masked, mask):
    """
    A_masked: (Nmask, nIC)
    mask: (H, W) bool
    Returns: maps_full (nIC, H, W)
    """
    H, W = mask.shape
    nIC  = A_masked.shape[1]
    maps_full = np.zeros((nIC, H*W), dtype=A_masked.dtype)
    mflat = mask.reshape(-1).astype(bool)
    maps_full[:, mflat] = A_masked.T
    return maps_full.reshape(nIC, H, W)

def split_mask_into_blobs(mask, connectivity=1, min_size=0):
    """
    mask: 2D binary array (0/1)
    connectivity: 1 = 4-connectivity, 2 = 8-connectivity
    min_size: remove blobs smaller than this many pixels

    Returns:
        list of 2D binary masks, one per blob
    """
    # Label connected components
    labeled = label(mask, connectivity=connectivity)

    blob_masks = []
    for region in regionprops(labeled):
        if region.area >= min_size:
            # Create mask for this blob
            blob = (labeled == region.label).astype(np.uint8)
            blob_masks.append(blob)

    return blob_masks

def plot_selected_components(selected_components,spatial_maps,nmf_mask_2d,nmf_mask_2d_smoothed, blobs):    
    """
        Plot selected NMF components along with their masks and smoothed masks.

        Parameters:
        - selected_components (list or iterable): Indices of components to visualize.
        - spatial_maps (array-like): 2D spatial maps for each component.
        - nmf_mask_2d (array-like): Binary or raw masks corresponding to each component.
        - nmf_mask_2d_smoothed (array-like): gaussian-filtered component masks.
        - blobs (list or dict): Blob detection results per component (used for counting blobs).

        Returns:
        - fig (matplotlib.figure.Figure): The generated figure containing the plots.
        """

    n_sel_comp=len(selected_components) 
    # More than one component -> create multiple rows (one per component)
    if n_sel_comp>1:
        fig,ax=plt.subplots(n_sel_comp,3, figsize=(9,3*n_sel_comp))
        for i, comp in enumerate(selected_components):
            ax[i,0].imshow(spatial_maps[comp],cmap='bwr')
            ax[i,0].set_title(f"Component {comp}, {len(blobs[comp])} blobs found")
            ax[i,1].imshow(nmf_mask_2d[comp])
            ax[i,2].imshow(nmf_mask_2d_smoothed[comp])
    #Exactly one component -> single row with 3 column
    elif n_sel_comp==1:
        fig,ax=plt.subplots(1,3, figsize=(9,3))
        for i, comp in enumerate(selected_components):
            ax[0].imshow(spatial_maps[comp],cmap='bwr')
            ax[0].set_title(f"Component {comp}, {len(blobs[comp])} blobs found")
            ax[1].imshow(nmf_mask_2d[comp])
            ax[2].imshow(nmf_mask_2d_smoothed[comp])
    # No components selected -> exit early
    else:
        return None
    for a in ax.flatten():
        a.axis('off')
    plt.tight_layout()
    return fig

def plot_components(to_plot):    
    """
    Plot a collection of 2D components (e.g., contours or masks).

    Parameters:
    - to_plot (array-like): Collection of 2D arrays to visualize.
      Shape is expected to be (n_components, height, width).

    Returns:
    - fig (matplotlib.figure.Figure): The generated figure.
    """

    n_sel_comp=to_plot.shape[0]
    
    # ---- Layout logic ----
    # If fewer than 7 components -> single row
    if n_sel_comp<7:
        fig, ax = plt.subplots(1,n_sel_comp,figsize=(n_sel_comp*2,2))
    else:
        # For larger sets -> multiple rows with 6 columns
        fig, ax = plt.subplots(int(np.ceil(n_sel_comp/6)), 6,figsize=(12,2*int(np.ceil(n_sel_comp/6))))
  
    if n_sel_comp==1:
        ax.imshow(to_plot[0])
        ax.set_title(f"Contour 0")
        ax.axis('off')
    else:
        ax=ax.flatten()
        for i, c in enumerate(to_plot):
            ax[i].imshow(c)
            ax[i].set_title(f"Contour {i}")
        for a in ax:
            a.axis('off')
    plt.tight_layout()
    return fig

def jaccard(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union != 0 else 0.0

def plot_clusters(to_plot_list, all_components):    
    """
    Plot a set of selected components along with their combined overlay.

    Parameters:
    - to_plot_list (list or iterable): Indices of components to display.
    - all_components (array-like): Collection of 2D component images.

    Behavior:
    - Displays each selected component in grayscale.
    - Adds one additional panel showing the sum (overlay) of all selected components.
    - Automatically adjusts subplot layout depending on how many panels are needed.
    """

    n = len(to_plot_list)
    n_total = n + 1  # +1 for overlay

    # ----- Layout logic -----
    # If few panels, use a single row
    if n_total < 7:
        fig, ax = plt.subplots(1, n_total, figsize=(n_total * 2, 2))
    else:
        # For larger numbers, arrange into multiple rows with 6 columns
        rows = int(np.ceil(n_total / 6))
        fig, ax = plt.subplots(rows, 6, figsize=(12, 2 * rows))

    ax = np.array(ax).flatten()

    # ---- Plot individual components ----
    for i, c in enumerate(to_plot_list):
        ax[i].imshow(all_components[c], cmap="gray")
        ax[i].axis("off")
        ax[i].set_title(f"component {c}")

    # ---- Overlay panel ----
    overlay = np.sum(
        [all_components[c] for c in to_plot_list],
        axis=0
    )

    # Plot the overlay in a heatmap style colormap
    ax[n].imshow(overlay, cmap="hot")
    ax[n].axis("off")
    ax[n].set_title("overlay")

    # Turn off unused axes
    for j in range(n + 1, len(ax)):
        ax[j].axis("off")

    plt.tight_layout()

def plot_contours_on_image(mean_image, all_contours, colors, pmin=0.1, pmax=99.9):
    """
    Plot contours over a grayscale image with discrete colors and legend.
    Parameters
    ----------
    mean_image : 2D array
        Background image shown in grayscale.
    all_contours : list
        List of contour groups. Each element is a list of contours,
        where each contour has shape (N, 2) with (x, y) coordinates.
    colors : array-like
        Array of RGB colors in [0, 1], shape (K, 3).        
   pmin, pmax : float
        Lower and upper percentiles used for contrast stretching.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.    """

    fig, ax = plt.subplots()   

    vmin = np.nanpercentile(mean_image, pmin)
    vmax = np.nanpercentile(mean_image, pmax) 

    n_contours = len(all_contours)
    ncol = 2 if n_contours > 17 else 1

    ax.imshow(mean_image, cmap="gray", vmin=vmin, vmax=vmax)
    for i, contours in enumerate(all_contours):
        color = colors[i % len(colors)]
        label_added = False
        for contour in contours:
            x = contour[:, 0]
            y = contour[:, 1]
            if not label_added:
                ax.plot(x, y, color=color, linewidth=2, label=str(i))
                label_added = True
            else:
                ax.plot(x, y, color=color, linewidth=2)
    ax.legend(
        title="Contour #",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
        ncol=ncol
    )
    ax.axis("off")
    fig.subplots_adjust(right=0.75)
    return fig

def bandpass_filter_traces(traces, fs, low=None, high=None, order=3):
    """
    Band-pass (or high/low-pass) filter for traces.

    Parameters
    ----------
    traces : ndarray, shape (n_traces, time)
        Each row is one trace
    fs : float
        Sampling frequency (Hz)
    low : float or None
        Low cutoff frequency (Hz)
    high : float or None
        High cutoff frequency (Hz)
    order : int
        Filter order

    Returns
    -------
    filtered : ndarray, shape (n_traces, time)    """

    if fs is None or (low is None and high is None):
        return traces
    nyq = fs / 2.0
    if low is not None and high is not None:
        sos = butter(order, [low / nyq, high / nyq],
                     btype="band", output="sos")
    elif low is not None:
        sos = butter(order, low / nyq,
                     btype="high", output="sos")
    else:
        sos = butter(order, high / nyq,
                     btype="low", output="sos")
    filtered = sosfiltfilt(sos, traces, axis=1)
    return filtered

def plot_traces(to_plot, spike_thresh):
    """
    Plot calcium traces with spike threshold and detected spike markers.

    Parameters
    ----------
    to_plot : np.ndarray
        Array of traces with shape (N_traces × T), where each row is a time series.
    spike_thresh : float
        Threshold value used to identify spikes in the traces.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plotted traces.
    """
    n_traces=to_plot.shape[0]
    if n_traces==1:
            fig,ax=plt.subplots(1,1, figsize=(12,3.5))
            ax.plot(to_plot[0])
            ax.plot((0,6000), (spike_thresh, spike_thresh), "--")
            ax.plot((0,6000), (0, 0), "k-")
            ax.plot(np.where(to_plot[0]>spike_thresh, to_plot[0], np.nan), "r*")
            ax.set_title(f"Contour 0")
    else:
        fig,ax=plt.subplots(n_traces,1, figsize=(12,3.5*n_traces))
        for i, trace in enumerate(to_plot):
            ax[i].plot(trace)
            ax[i].plot((0,6000), (spike_thresh, spike_thresh), "--")
            ax[i].plot((0,6000), (0, 0), "k-")
            ax[i].plot(np.where(trace>spike_thresh, trace, np.nan), "r*")
            ax[i].set_title(f"Contour {i}")

    return fig

def find_segments(signal, thresh, sign):        
    """
    Identify contiguous segments where a signal crosses a given threshold.
    Returns segments as arrays of [values, indices].
    """
    temp=[] # collect signal values within a segment
    itemp=[] # collect corresponding indices
    segments=[] # list of detected segments
    flag=False # indicates whether currently inside a segment
    # Define threshold condition
    if sign=='>':
        crit=signal>thresh
    elif sign=='<':
        crit=signal<thresh
    else:
        raise ValueError('Unknown sign')
    # Iterate through signal to group contiguous points
    for i in range(len(signal)):
        if crit[i]:
            temp.append(signal[i])
            itemp.append(i)
            flag=True
        else:
            if flag:
                segments.append(np.array([temp,itemp]))
                temp=[]
                itemp=[]
                flag=False
    # Keep only segments longer than 1 point
    segments=[s for s in segments if len(s[0])>1]
    return segments    

def get_peaks_maxsegments(traces, precentile_threshold=80, strict=3, dF_thresh=0.05):
    """
    Detect calcium activity peaks and their amplitudes using a threshold-based segmentation approach.

    Parameters
    ----------
    traces : np.ndarray
        Array of shape (N_ROIs x T), containing ΔF/F traces for each ROI.
    precentile_threshold : float, optional (default=80)
        Percentile used to separate baseline from active signal for each trace.
    strict : float, optional (default=3)
        Multiplier of standard deviation used to define the peak detection threshold.
    dF_thresh : float, optional (default=0.05)
        Minimum ΔF/F amplitude required for a detected peak to be kept.

    Returns
    -------
    all_peaks : dict
        Dictionary mapping ROI index -> list of (start, max) tuples indicating
        detected peaks (start index and peak maximum index).
    all_amp : dict
        Dictionary mapping ROI index -> list of peak amplitudes (ΔF/F at peak max).

    """
    # Initialize output dictionarie
    all_peaks={}
    all_amp={}
    #Estimate baseline using percentile
    pers=np.percentile(traces, precentile_threshold, axis=1)
    pers=np.reshape(pers,(len(pers),1))
    # Keep only values below percentile (baseline region), set others to NaN
    filtered=np.where(traces<pers, traces, np.nan)
    # Compute baseline mean and standard deviation per R
    mn=np.nanmean(filtered, axis=1)
    sd=np.nanstd(filtered,axis=1)
    # Process each ROI independently
    for roi in range(traces.shape[0]):
        # Find segments where signal exceeds dynamic threshold (mean + strict * SD)
        segmax=find_segments(traces[roi], mn[roi]+strict*sd[roi], '>')
        # Find peak maxima indices within each segment
        mxi=[] #indices of maximums      
        for s in segmax:
            mx=max(s[0]) # maximum value in segment
            for i in range(len(s[1])):
                if s[0][i]==mx:
                    mxi.append(int(s[1][i]))
        #Backtrack to find peak start indices    
        mni=[] # indices of peak starts
        for mi in mxi:
            x=mi
            # Move backward until signal falls to near baselin
            while x>0 and traces[roi][x]>mn[roi]+sd[roi]:
                x-=1
                # If another peak is encountered, adjust start to local minimum
                if x in mxi:
                    x=x+np.argmin(traces[roi][x:mi])
                    break
            mni.append(x)
        #Filter peaks and store results
        peaks=[]
        amp=[]
        for i in range(len(mxi)):
            # Apply amplitude threshold
            if traces[roi][mxi[i]]>dF_thresh:
                peaks.append((mni[i],mxi[i]))
                amp.append(traces[roi][mxi[i]])
        # Store results for this ROI
        all_peaks[roi]=peaks
        all_amp[roi]=amp
    return all_peaks, all_amp

def plot_traces_peaks(to_plot, peaks, spike_thresh):
    """
    Plot calcium traces with detected peaks (start and maximum) and threshold.

    Parameters
    ----------
    to_plot : np.ndarray
        Array of traces with shape (N_traces x T), where each row is a time series.
    peaks : dict
        Dictionary mapping ROI index -> list of (start_idx, max_idx) tuples 
        representing detected peaks.
    spike_thresh : float
        Threshold used for peak detection (visualized as dashed line).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plotted traces and annotated peaks.
    """
    n_traces=to_plot.shape[0]
    if n_traces==1:
            fig,ax=plt.subplots(1,1, figsize=(12,3.5))
            ax.plot(to_plot[0])
            ax.plot((0,6000), (spike_thresh, spike_thresh), "--")
            ax.plot((0,6000), (0, 0), "k-")
            for peak in peaks[0]:
                ax.plot(peak[0],to_plot[0][peak[0]],'oy')
                ax.plot(peak[1],to_plot[0][peak[1]],'or')
            ax.set_title(f"Contour 0")
    else:
        fig,ax=plt.subplots(n_traces,1, figsize=(12,3.5*n_traces))
        for i, trace in enumerate(to_plot):
            ax[i].plot(trace)
            ax[i].plot((0,6000), (spike_thresh, spike_thresh), "--")
            ax[i].plot((0,6000), (0, 0), "k-")
            for peak in peaks[i]:
                ax[i].plot(peak[0],trace[peak[0]],'oy')
                ax[i].plot(peak[1],trace[peak[1]],'or')
            ax[i].set_title(f"Contour {i}")

    return fig


def plot_shaded_histogram(to_plot, n_bins=10, xlabel="Value", ylabel="Count", title="Histogram (mean ± SD across groups)", ax=None, label=None):
    """
    Plot an averaged histogram across ROIs with standard deviation shading.

    Parameters
    ----------
    to_plot : np.ndarray
        Array of shape (N_ROIs × N_values_per_ROI), containing data to histogram.
    n_bins : int, optional (default=10)
        Number of bins for the histogram.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, a new figure and axis are created.
    label : str, optional
        Label for the plotted mean line (useful for legends).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axis containing the plot.
    """

    # --- Create ax if not provided ---
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    roi_hists = [] # store histogram counts for each ROI

    # Compute global bin edges across all ROIs
    all_data = []
    for roi in range(to_plot.shape[0]):
        d = to_plot[roi]
        d = d[np.isfinite(d)] # remove NaN/inf
        if d.size > 0:
            all_data.append(d)
    if len(all_data) == 0:
        raise ValueError("No valid data to plot.")
    # Concatenate all ROI data and define shared bin
    all_data = np.concatenate(all_data)
    bins = np.histogram_bin_edges(all_data, bins=n_bins)

    # Compute histogram for each ROI
    for roi in range(to_plot.shape[0]):
        data = to_plot[roi]
        data = data[np.isfinite(data)]
        if data.size == 0:
            # If no valid data, fill with NaNs
            roi_hists.append(np.full(len(bins) - 1, np.nan))
            continue
        counts, _ = np.histogram(data, bins=bins)
        roi_hists.append(counts)
    # Stack histograms -> shape (N_ROIs × N_bins)
    roi_hists = np.vstack(roi_hists)
    # Compute mean and SD across ROIs
    mean_hist = np.nanmean(roi_hists, axis=0)
    sd_hist = np.nanstd(roi_hists, axis=0)
    # Compute bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # lot mean histogram and SD shading
    ax.plot(bin_centers, mean_hist, lw=2,color="k", label=label)
    ax.fill_between(
        bin_centers,
        mean_hist - sd_hist,
        mean_hist + sd_hist,
        color="k",
        alpha=0.3
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if label is not None:
        ax.legend()

    return fig, ax

def plot_raster(peaks_max, rois='all', nframes=6000, fs=10,  title="Raster plot"):    

    """
    Generate a raster plot of peak timings across ROIs.

    Parameters
    ----------
    peaks_max : np.ndarray or array-like
        Peak maxima indices for each ROI (per ROI list/array of frame indices).
    rois : list or 'all', optional (default='all')
        Subset of ROIs to include. If 'all', all ROIs are plotted.
    nframes : int, optional (default=6000)
        Total number of frames for the raster (time axis length).
    fs : float, optional (default=10)
        Sampling frequency in Hz (used to convert frames -> seconds).
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the raster plot.
    raster : np.ndarray
        Binary matrix of shape (nframes × N_ROIs), where 1 indicates a peak event.

    """
    #Select ROIs
    if rois=='all':
        rois=list(range(peaks_max.shape[0]))
    #Initialize raster matrix (time x ROIs)
    raster=np.zeros((nframes, len(rois)))
    #Fill raster with peak events
    for roi in rois:
        # Extract valid peak indices and convert to integers
        mx = peaks_max[roi][np.isfinite(peaks_max[roi])].astype(int)
        raster[mx,roi]=1 # mark peak occurrences
    
    #Create raster plot
    fig, ax = plt.subplots(1,1,figsize=[15,len(rois)/2.5])
    for roi in rois:
        # Convert frame indices to time in seconds
        ind=[i*(1/fs) for i,e in enumerate(raster[:,roi]>0) if e]
        # Plot each event as a dot at (time, ROI index
        ax.plot(ind,[roi]*len(ind),'.k')
        ax.set_xlabel('time (sec)', fontsize=16)
        ax.set_ylabel('Contour #', fontsize=16)
        ax.set_title(title, fontsize=16)
        # Grid and tick formatting
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.grid(axis="x", which="minor", linestyle="--", alpha=0.5)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(12)
    return fig, raster

def plot_population_activity (raster, nframes=6000, fs=10,  title="Population activity plot"):    
    """
    Plot population activity over time based on a raster matrix.

    Parameters
    ----------
    raster : np.ndarray
        Binary matrix of shape (T x N_ROIs), where 1 indicates activity (peak)
        and 0 indicates inactivity.
    nframes : int, optional (default=6000)
        Number of frames (time points) to display.
    fs : float, optional (default=10)
        Sampling frequency in Hz (used to convert frames -> seconds).
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the population activity plot.

    Notes
    -----
    - Population activity represents the global network activity,
    defined as the number of ROIs active at each time point.
    - Effectively, this is the sum of the raster plot across ROIs.
    - The horizontal dashed lines indicate:
        * Red line: at least one active ROI
        * Yellow line: ≥50% of ROIs active (proxy for synchrony)
    """
    #Compute population activity
    # Sum across ROIs -> number of active components per time point
    to_plot = np.sum(raster, axis=1)
    # Threshold for 50% of ROIs (synchrony reference)
    thresh_50=raster.shape[1]//2

    #Plot population activity
    fig, ax = plt.subplots(1,1,figsize=[15,6])
    ind=[i*(1/fs) for i in range (nframes)] # create time axis
    ax.plot(ind,to_plot)
    # Add reference thresholds
    ax.plot((ind[0], ind[-1]), (1,1), "r--")
    ax.plot((ind[0], ind[-1]), (thresh_50,thresh_50), "y--")
    # Formatting
    ax.set_xlabel('time (sec)', fontsize=16)
    ax.set_ylabel('Number of active contours', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis="x", which="minor", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(12)
    return fig

def plot_peak_heatmap(peaks_norm, Z, clusters, fast_clust, fr, fs):
    """
    Plot a clustered heatmap of normalized calcium peaks with cluster annotation.

    Parameters
    ----------
    peaks_norm : np.ndarray
        2D array (N_peaks x T), normalized peak waveforms.
    Z : np.ndarray
        Linkage matrix for hierarchical clustering (e.g., from scipy.cluster.hierarchy).
    clusters : array-like
        Cluster assignment for each peak.
    fast_clust : int
        Identifier for the cluster corresponding to "fast" peaks.
    fr : int
        Number of frames in each peak window.
    fs : float
        Sampling frequency (Hz) to convert frames -> seconds.

    Returns
    -------
    g : seaborn.matrix.ClusterGrid
        Seaborn clustermap object containing heatmap and dendrogram.

    Notes
    -----
    - Rows are clustered, columns (time) are not.
    - Row colors indicate cluster identity (fast vs slow peaks).
    - Heatmap intensity represents normalized signal amplitude (z-score).
    """

    # Define colors for clusters
    palette = ["#C44E52", "#6C757D"]  # red-like = fast, gray = slow
    row_colors = [palette[0] if c == fast_clust else palette[1] for c in clusters]

    # Create clustered heatmap
    g = sns.clustermap(
        peaks_norm,
        row_linkage=Z,           # hierarchical clustering order
        row_colors=row_colors,   # color-code rows by cluster
        col_cluster=False,       # do not cluster time axis
        yticklabels=False,       # hide row labels
        cmap="viridis",          # color map for intensity
        figsize=(peaks_norm.shape[1]/2.5, peaks_norm.shape[0]/5),
        dendrogram_ratio=(0.2, 0.0),  # show only row dendrogram
        cbar_kws={"label": "Intensity (z-score)"},
        cbar_pos=(1.01, 0.6, 0.05, 0.18)  # colorbar position
    )

    # Add legend for cluster labels
    legend_elements = [
        Patch(facecolor=palette[0], label="Fast peaks"),
        Patch(facecolor=palette[1], label="Slow peaks")
    ]

    g.ax_heatmap.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.5, 1)
    )

    # Convert x-axis from frames to seconds
    new_labels = [round(i / fs, 2) for i in range(fr + 1)]
    g.ax_heatmap.set_xticks(range(len(new_labels)))
    g.ax_heatmap.set_xticklabels(new_labels)

    # Labels and title
    g.ax_heatmap.set_xlabel("Time (s)")
    g.fig.suptitle(r"Ca$^{2+}$ peaks clustered by shape", y=1.02)

    return g

def compute_histogram_df(time_to_peak, bin_start=0, bin_end=5, step=0.2):
    """
    Compute histogram (per ROI) for time-to-peak data.

    Parameters:
        time_to_peak : 2D numpy array (ROIs x peaks), values in frames
        bin_start : int, start of bins (default 0)
        bin_end : int, end of bins (default 50)
        step : int, bin width (default 2)

    Returns:
        pandas DataFrame with:
            - rows: bin centers
            - columns: ROI indices
            - values: histogram counts (NaN if ROI has no data)
    """
    # Define fixed bin edges and centers
    bins = np.arange(bin_start, bin_end + step, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    roi_hists = [] # store histogram for each ROI

    # Compute histogram per ROI
    for roi in range(time_to_peak.shape[0]):
        data = time_to_peak[roi]
        data = data[np.isfinite(data)] # Remove NaN and infinite values

        if data.size == 0:
            # preserve missing ROI as NaNs
            roi_hists.append(np.full(len(bins) - 1, np.nan))
        else:
            # Compute histogram count
            counts, _ = np.histogram(data, bins=bins)
            # Normalize to probability distribution
            counts = counts / counts.sum()
            roi_hists.append(counts.astype(float))  # ensure consistent dtype

    # Stack histograms → shape (N_ROIs × N_bin
    roi_hists = np.vstack(roi_hists)

    # Create DataFrame (bins as rows, ROIs as columns) 
    df = pd.DataFrame(
        roi_hists.T,  # transpose: bins as rows, ROIs as columns
        index=bin_centers
    )
    df.index.name = "bin_center"
    df.columns = [f"ROI_{i}" for i in range(df.shape[1])]

    return df.mean(axis=1)
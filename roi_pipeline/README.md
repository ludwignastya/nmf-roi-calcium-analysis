# ROI-Based Calcium Imaging Analysis Pipeline
This repository contains a structured workflow for analyzing calcium imaging data using a region-of-interest (ROI)–based approach, focused on extracting pixel-wise activity, detecting calcium events, and quantifying network dynamics.

## Repository Structure

### Core utilities:
- TracesMaskedNew.py
- ROI extraction from masked calcium imaging movies (pixel-wise ROIs)
- Signal preprocessing and normalization
- High-pass, low-pass, and bandpass filtering
- Peak detection using adaptive thresholding
- Event segmentation (start, peak, duration)
- Raster construction
- Inter-event interval computation
- ROI activity and network metrics
- Correlation analysis and visualization tools

### Analysis Pipeline (scripts / workflow)
- Mask-based ROI extraction
- Spatial downsampling (square ROI definition)
- Trace extraction
- Signal filtering 
- Peak detection and segmentation
- ROI activity classification
- Raster construction 
- Temporal analysis (frequency, intervals)
- Network activity computation (active area over time)
- Event classification (local, medium, global)
- Statistical summary and export

### Data Flow
Raw movie + mask → ROI extraction → filtering → traces → peaks → raster → activity analysis → statistics

### Outputs
Traces (ΔF/F-like normalized signals)
Detected peaks
Raster plots 
ROI activity frequency
Inter-event intervals
Network activity (active area over time)
Event statistics (local / medium / global classification)
Correlation matrices
Figures (.png) and numerical outputs (.txt, .xlsx)

### Requirements
Python 3.11
numpy 2.4.2
pandas 3.0.1
scipy 1.17.1
matplotlib 3.10.8
tifffile 2026.2.24
skimage 0.26.0

### Usage
Run the analysis script step-by-step in the provided notebook file.

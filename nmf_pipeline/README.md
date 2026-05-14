# NMF-Based Calcium Imaging Analysis Pipeline

This repository contains a modular workflow for analyzing calcium imaging data using Non-negative Matrix Factorization (NMF), followed by component refinement, signal extraction, peak detection, and population-level analysis.

## Repository Structure

### Core utilities:
nmf_utils.py
- ΔF/F (dF/F) computation and preprocessing
- NMF component handling and spatial map reconstruction
- Mask generation, refinement, and blob splitting
- Component similarity analysis (e.g., Jaccard index)
- Signal filtering (bandpass) and trace processing
- Peak detection, segmentation, and feature extraction
- Population and network activity analysis (raster, population trace)
- Histogram and statistical aggregation utilities
- Visualization tools (traces, masks, raster plots, heatmaps, histograms, contours)

### Analysis Pipeline (notebooks)
1. NMF decomposition
2. Component splitting and merging
3. Manual correction
4. Visualization of components
5. Area statistics
6. Extract traces
7. Peak detection
8. Peak statistics
9. Network activity analysis
10. Peak shape clustering
11. Time-to-peak histograms

### Data Flow
Raw movie + mask → dF/F → NMF → refinement → traces → peaks → analysis

### Outputs
- Spatial maps
- Traces
- Peaks
- Raster plots
- Population activity
- Heatmaps
- Statistics (.xlsx)

### Requirements
Python 3.11
numpy 2.4.2
pandas 3.0.1
scipy 1.17.1
sklearn 1.8.0
matplotlib 3.10.8
seaborn 0.13.2
tifffile 2026.2.24
skimage 0.26.0

Usage
Run notebooks sequentially from 1 to 11.

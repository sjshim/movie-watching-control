# plot_utils.py
# Purpose: Plotter functions

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

from nonparametric import null_threshold
from tools import get_factors, count_nans

def transform_and_plot(transform_func, plotter_func):
    """Small closure function for any nilearn plotting function 
    that automatically inverse transforms array data back into Nifti1Image
    using a functon that transform data from array to an Nifti1Image object 
    before plotting results.
    
    eg.,
    fig, ax = plt.subplots()
    custom_stat_map = transform_and_plot(gm_masker.inverse_transform, plot_stat_map)
    custom_stat_map(observed_stats, threshold=0.2, axes=ax)
    plt.show()
    """
    def plotter(vox_stats, *stat_map_args, **stat_map_kwargs):
        transformed_data = transform_func(vox_stats)
        return plotter_func(transformed_data, *stat_map_args, **stat_map_kwargs)
    return plotter

def get_subplot_grid(n_plots, grid_shape='squarish'):
    """
    Get subplot row and column sizes based on the total number
    of subplots desired.
    
    grid_shape can be either 'squarish', 'landscape', or 'portrait'
    """
    
    assert grid_shape in ('squarish', 'landscape', 'portrait'), "grid_shape must be 'squarish', 'landscape', or 'portrait' sorry lol"
    
    # make n_plots an even number
    if n_plots % 2 != 0:
        n_plots += 1
    factors = get_factors(n_plots)
    n_factors = len(factors)
    
    # Choose middle factors or ending factors depending on chosen grid shape
    if grid_shape == 'squarish':
        n_rows, n_cols = factors[(n_factors//2)-1 : (n_factors//2)+1]
    elif grid_shape == 'landscape': # aka n_rows < n_cols
        n_rows, n_cols = factors[1], factors[-2]
    elif grid_shape == 'portrait':
        n_rows, n_cols = factors[-2], factors[1]
    
    return n_rows, n_cols

def plot_thresholded_func(data=None, null_dist=None, 
                          threshold_data=None,
                          masker=None, 
                          alpha=0.05,
                          max_stat=False,
                          axes=None, 
                          cmap='RdBu_r',
                          threshold=None,
                          **plotter_kw):
    
    if threshold_data is None:
        threshold_data = null_threshold(data, null_dist=null_dist, 
                                        alpha=alpha, max_stat=max_stat)
    if threshold is None:
        threshold = np.nanmin(threshold_data)
        
    threshold_data = masker.inverse_transform(threshold_data)
    plot_stat_map(threshold_data, 
                  axes=axes,
                  cmap=cmap,
                  threshold=threshold,
                  **plotter_kw)


# Plots null distribution and handles odd number of subplots required
def plot_null_dist_hist(null_dist, grid_shape='squarish', bins=None, title=None, figsize=None, 
                   layout_kw={'pad':1.08, 'h_pad':None, 'w_pad':None, 'rect':None}):
    
    """
    Plots provided null distribution and allows for plotting an odd number of subplots
    """
    
    # Add one plot to account for the entire null distribution histogram
    n_plots = null_dist.shape[0] + 1
    n_rows, n_cols = get_subplot_grid(n_plots, grid_shape=grid_shape)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"shape: {null_dist.shape}")
        
    # Plot entire distribution in first subplot
    ax[0,0].hist(null_dist.ravel(), bins=bins)
    ax[0,0].set_title(f"Entire null distribution\ndata shape: {null_dist.ravel().shape}")
    
    # Plot each null distribution iteration in a seprate subplot
    for i in range(null_dist.shape[0]):
        perm_i = null_dist[i]
        ax.ravel()[i+1].hist(perm_i, bins=bins)
        ax.ravel()[i+1].set_title(f"iteration {i}\ndata shape: {perm_i.shape}\nnan count: {count_nans(perm_i)}")

    # Delete remaining subplots from main figure
    n_extra_plots = n_rows*n_cols - n_plots
    for i in range(n_extra_plots):
        fig.delaxes(ax.ravel()[-(i+1)])
    fig.tight_layout(**layout_kw)
    plt.show()
    
def plot_hist_vline(hist_data, vline_data, bins=None, ax=None, line_color=['r'], **vline_kw):
    if ax is None:
        ax = plt
    ax.hist(hist_data, bins=bins)
    for i, line in enumerate(vline_data):
        ax.axvline(vline_data, color=line_color[i], **vline_kw)
    ax.axvline(np.nanmean(hist_data), color='black', label=f'mean={hist_data.mean():.3f}')
    ax.legend()
    
def plot_null_hist_with_threshold(hist_data, alpha, tail='upper', bins=None, ax=None):
    assert tail in ('upper', 'lower', 'two-sided'), "tail must be 'upper', 'lower', or 'two-sided'"
    if ax is None:
        ax = plt
    
    percentile = lambda alpha: np.nanpercentile(hist_data, alpha)
    
    if tail=='upper':
        upper_threshold = percentile((1.0-alpha)*100)
        args = [hist_data, [upper_threshold]]
        kwargs = dict(line_color=['r'], bins=bins, ax=ax,
                      label=f'upper={upper_threshold:.3f} (alpha={alpha})')
    elif tail == 'lower':
        lower_threshold = percentile(alpha*100)
        args = [hist_data, [lower_threshold]]
        kwargs = dict(line_color=['y'], bins=bins, ax=ax,
                      label=f'lower={lower_threshold:.3f} (alpha={alpha})')
    elif tail == 'two-sided':
        upper_threshold = percentile((1-alpha)*100)
        lower_threshold = percentile(alpha*100)
        args = [hist_data, [upper_threshold, lower_threshold]]
        label = [f'upper={upper_threshold:.3f}', f'lower={lower_threshold:.3f}']
        kwargs = dict(line_color=['r', 'y'], bins=bins, ax=ax, label=label)
    plot_hist_vline(*args, **kwargs)
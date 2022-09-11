# intersubject.py: 
# Purpose: Contains isc functions that expand on the designs in 
# brainiak's isc module.

from functools import partial
from datetime import timedelta
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nibabel import Nifti1Image
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial.distance import squareform
from .utils.bnk_funcs import (array_correlation, _check_timeseries_input,
                            _threshold_nans, compute_summary_statistic)

logger = logging.getLogger(__name__)

# ===================
# Basic ISC functions
# ===================


def isc(data, pairwise=False, 
        summary_statistic=None, 
        tolerate_nans=True, 
        n_jobs=None,
        joblib_kwargs={}):
    """
    Calculate leave-one-out (loo-ISC) or pairwise ISC.
        loo-ISC: for each brain region, correlate the subject i's timeseries
        with the group average-minus-subject-i's timeseries.

        pairwise-ISC: for each brain region, compute the pairwise correlation
        matrix between every pair of subject's timeseries.

    Parameters
    ----------
    data: 3d array of shape (n_TRs, n_regions, n_subjects)
        An array of individual-level brain timeseries data
    
    pairwise: bool, default=False
        If True, compute pairwise ISC. Otherwise, leave-one-out ISC is
        computed by default.

    summary_statistic: str, default=None
        Choose to average ISC results using 'mean' or 'median'. ISC values
        are automatically Fisher Z transformd before being averaged and
        inverse Fisher Z transformed back as r-values.

    tolerate_nans: bool, default=True
        The proportion of subjects with non-NaN values required to keep
        voxel.

    n_jobs: int, default=None
        Number of processers to devote to performing computation
        in parallel. If None, then a normal for loop with be used;
        if -1, then all processors will be used to parallelize computation. 

        This function extends the Brainiak ISC implementation with 
        Joblib parallelisation.
        
    Returns
    -------
    array, where shape is:
        (n_brain_region_iscs, n_subjects) if pairwise=False
        (n_brain_region_iscs, n_subject_pairs) if pairwise=True
        (n_brain_region_iscs) if summary_statistic=True
    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # No summary statistic if only two subjects
    if n_subjects == 2:
        logger.info("Only two subjects! Simply computing Pearson correlation.")
        summary_statistic = None

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    data, mask = _threshold_nans(data, tolerate_nans)

    # Compute correlation for only two participants
    if n_subjects == 2:

        # Compute correlation for each corresponding voxel
        iscs_stack = array_correlation(data[..., 0],
                                       data[..., 1])[np.newaxis, :]

    # Compute pairwise ISCs using voxel loop and corrcoef for speed
    elif pairwise:

        # Swap axes for np.corrcoef
        data = np.swapaxes(data, 2, 0)
      
        pairwise_corr = lambda x: squareform(np.corrcoef(x), checks=False)

        # Loop through voxels
        if n_jobs is not None:
            with Parallel(n_jobs=n_jobs, **joblib_kwargs) as parallel:
                voxel_iscs = parallel(delayed(pairwise_corr)(data[:,v,:])
                                    for v in range(data.shape[1]))
        else:
            voxel_iscs = []
            for v in np.arange(data.shape[1]):
                voxel_data = data[:, v, :]

                # Correlation matrix for all pairs of subjects (triangle)
                iscs = squareform(np.corrcoef(voxel_data), checks=False)
                voxel_iscs.append(iscs)

        iscs_stack = np.column_stack(voxel_iscs)

    # Compute leave-one-out ISCs
    elif not pairwise:
        loo_corr = lambda x, s: array_correlation(
                                    x[...,s],
                                    mean(np.delete(x, s, axis=2), axis=2))
        if n_jobs is not None:
            with Parallel(n_jobs=n_jobs, **joblib_kwargs) as parallel:
                iscs_stack = parallel(delayed(loo_corr)(data, s)
                                    for s in range(n_subjects))
        else:
            iscs_stack = [loo_corr(data, s) for s in range(n_subjects)]
            
        iscs_stack = np.array(iscs_stack)
#         iscs_stack = np.array(loo_corr(data, tolerate_nans, n_jobs))

    # Get ISCs back into correct shape after masking out NaNs
    iscs = np.full((iscs_stack.shape[0], n_voxels), np.nan)
    iscs[:, np.where(mask)[0]] = iscs_stack

    # Summarize results (if requested)
    if summary_statistic:
        iscs = compute_summary_statistic(iscs,
                                         summary_statistic=summary_statistic,
                                         axis=0)[np.newaxis, :]

    # Throw away first dimension if singleton
    if iscs.shape[0] == 1:
        iscs = iscs[0]

    # return iscs
    return iscs.T


def wmb_isc(d1, d2, subtract_wmb=False, summary_statistic=None, 
            tolerate_nans=True, n_jobs=None, joblib_kwargs={}):
    """
    Compute within-minus-between ISC between two group of subjects' timeseries
    data. This is calculate for each subject by calculating within-group ISC
    (subject correlated with the within-group average) and between-group ISC
    (subject correlated with the group-minus-subject-i average) for each brain 
    region. Each subject's own within-group and between-group ISCs are finally
    subtracted from one another to obtain their within-minus-between ISC.

    Parameters
    ----------
    d1, d2: 3d arrays of shape (n_TRs, n_regions, n_subjects) for two
        groups of subjects's timeseries data.
    
    pairwise: bool, default=False
        If True, compute pairwise ISC. Otherwise, leave-one-out ISC is
        computed by default.

    summary_statistic: str, default=None
        Choose to average ISC results using 'mean' or 'median'. ISC values
        are automatically Fisher Z transformd before being averaged and
        inverse Fisher Z transformed back as r-values.

    tolerate_nans: bool, default=True
        The proportion of subjects with non-NaN values required to keep
        voxel.

    n_jobs: int, default=None
        Number of processers to devote to performing computation
        in parallel. If None, then a normal for loop with be used;
        if -1, then all processors will be used to parallelize computation. 

        This function extends the Brainiak ISC implementation with 
        Joblib parallelisation.
        
    Returns
    -------
    array, where shape is:
        (n_brain_region_iscs, n_subjects, within_and_between_group) if subtract_wmb=False
        (n_brain_region_wmb_iscs, n_subjects) if subtract_wmb=True
        (n_brain_region_wmb_iscs) if subtract_wmb=True and summary_statistic=True


    """
    
    # assert d1.shape == d2.shape, "d1 and d2 must have equal shapes"
    d1, d1_n_TRs, d1_n_voxels, d1_n_subs = _check_timeseries_input(d1)
    d2, d2_n_TRs, d2_n_voxels, d2_n_subs = _check_timeseries_input(d2)
    
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    if summary_statistic=='mean':
        summary_statistic = np.nanmean
    elif summary_statistic=='median':
        summary_statistic = np.nanmedian

    d1_and_d2 = np.append(d1, d2, axis=-1)
    d1_and_d2, mask = _threshold_nans(d1_and_d2, tolerate_nans)
    d1 = d1_and_d2[..., : d1_n_subs]
    d2 = d1_and_d2[..., d1_n_subs :]
    del d1_and_d2
    # d2, d2_mask = _threshold_nans(d2, tolerate_nans)
    
    # Calculate within and between group isc for each group separately, then append
    loo_corr = lambda x, s: array_correlation(
                                x[...,s],
                                mean(np.delete(x, s, axis=2), axis=2))
    one2avg_corr = lambda x_i, y: array_correlation(
                                    x_i, 
                                    mean(y, axis=2), axis=2)
    
    w_iscs_stack = []
    b_iscs_stack = []
    data_tup = (d1, d2)
    if n_jobs is not None:
        with Parallel(n_jobs=n_jobs, **joblib_kwargs) as parallel:
            for idx, d in enumerate(data_tup):
                n_subjects = data_tup[idx].shape[-1]
                w_iscs_stack += parallel(delayed(loo_corr)(data_tup[idx], s)
                                        for s in range(n_subjects))
                b_iscs_stack += parallel(n_jobs=n_jobs)(delayed(one2avg_corr)(data_tup[idx][...,s], data_tup[idx-1])
                                        for s in range(n_subjects))
                    
    else:
        for idx, d in enumerate(data_tup):
            n_subjects = data_tup[idx].shape[-1]
            for s in range(n_subjects):
                w_iscs_stack.append(loo_corr(data_tup[idx], s))
                
                b_iscs_stack.append(one2avg_corr(data_tup[idx][...,s], 
                                                data_tup[idx-1]))
    
    w_iscs_stack, b_iscs_stack = np.array(w_iscs_stack), np.array(b_iscs_stack)
    
    # Get original data shape after masking out NaNs
    within_isc = np.full((w_iscs_stack.shape[0], d1_n_voxels), np.nan)
    between_isc = np.full((b_iscs_stack.shape[0], d1_n_voxels), np.nan)
    within_isc[:, np.where(mask)[0]] = w_iscs_stack
    between_isc[:, np.where(mask)[0]] = b_iscs_stack

    wmb_iscs = np.array([within_isc, between_isc])
    if subtract_wmb:
        wmb_iscs = within_isc - between_isc
        if summary_statistic:
            # wmb_iscs = compute_summary_statistic(wmb_iscs,
            #                                 summary_statistic=summary_statistic,
            #                                 axis=0)[np.newaxis, :].squeeze(axis=0)
            wmb_iscs = summary_statistic(wmb_iscs, axis=0)
            
        # return wmb_iscs
        return wmb_iscs.T
    else:
        if summary_statistic:
            # wmb_iscs = compute_summary_statistic(wmb_iscs,
            #                                 summary_statistic=summary_statistic,
            #                                 axis=1)[np.newaxis, :].squeeze(axis=0)
            wmb_iscs = summary_statistic(wmb_iscs, axis=1)
        return wmb_iscs.T

# ===========================
# Movie-segment ISC functions
# ===========================

def dynamic_func(func, d1=None, d2=None, window_size=5, step=1, gaussian_filter_mode='reflect', 
                    sigma=3, filter_kwargs={}, func_kwargs={}):
    logger.debug(f"Running dynamic_func(window_size={window_size}, step={step})")
    if d2 is not None:
        assert len(d1)==len(d2), f"d1 and d2 length must be equal, but they were {len(d1)} and {len(d2)} instead"
    
    out = []
    for window in window_generator(d1, d2=d2, window_size=window_size, step=step, start=0):
        if d2 is not None:
            d1, d2 = window
        else:
            d1 = window
        if gaussian_filter_mode:
            if d2 is not None:
                d2 = gaussian_filter1d(d2, sigma, axis=0,
                                mode=gaussian_filter_mode,
                                **filter_kwargs)
            d1 = gaussian_filter1d(d1, sigma, axis=0, mode=gaussian_filter_mode,
                    **filter_kwargs)
        if d2 is not None:
            res = func(d1, d2, **func_kwargs)
        else:
            res = func(d1, **func_kwargs)
        out.append(res)
    out = np.array(out)
    return out


def window_generator(d1, d2=None, window_size=None, start=0, step=1, stop=None):
    assert window_size is not None, f"window_size must be an integer and cannot be None"
    # if d2 is not None:
    #     assert len(d1) == len(d2), f"len of d1 and d2 must be equal, but were {len(d1)} and {len(d2)} instead"
    assert start > -1, f"start must be a positive integer"
    if stop is not None:
        assert stop > 0 and stop > start, f"stop must be greater than 0 and start"
    if stop is None:
        stop = len(d1)

    # assert stop is not None, f"stop must be an integer larger than start"
    if d2 is not None:
        assert len(d1)==len(d2), f"d1 and d2 length must be equal, but they were {len(d1)} and {len(d2)} instead"
    # try:
    while start+window_size <= stop:
        logger.debug(f"start={start}, stop={stop}, step={step}")
#             print(f"start={start},stop={stop},step={step}")
        sl = slice(start, start+window_size)
        if d2 is not None:
            yield (d1[sl], d2[sl])
        else:
            yield d1[sl]
        start += step
        # stop += step  
    # except:
    #     pass


def isc_by_segment(data, seg_trs, method='loo', summary_statistic=None, tolerate_nans=True, 
                subtract_wmb=False, n_jobs=None):
    """
    Compute an ISC variant repeatedly over every n segment of trs. 
    This will return ISC results in a new array where axis=0's size is equal to
    data.shape[0] / seg_trs.
    """

    try:         
        if method != 'wmb':
            assert data.shape[0] % seg_trs == 0
            n_TRs = data.shape[0]
        else:
            assert data[0].shape[0] % seg_trs == 0
            n_TRs = data[0].shape[0]
    except:
        raise "data TR length be divisble by seg_trs with no remainder."
    n_segments = int(n_TRs / seg_trs)
    seg_idx = n_segments
    segment_isc = []
    
    if method == 'loo':
        isc_func = partial(isc, pairwise=False, 
                           summary_statistic=summary_statistic, 
                           tolerate_nans=tolerate_nans, n_jobs=n_jobs)
    elif method == 'pairwise':
        isc_func = partial(isc, pairwise=True, 
                           summary_statistic=summary_statistic, 
                           tolerate_nans=tolerate_nans, n_jobs=n_jobs)
    elif method == 'wmb': # currently assumes wmb is leave one out isc-based
        assert type(data) is list, "data must be list of two group's data for wmb isc"
        isc_func = partial(wmb_isc, subtract_wmb=subtract_wmb, tolerate_nans=tolerate_nans,
                          n_jobs=n_jobs)

    while seg_idx > 0:
        start = (n_segments - seg_idx) * seg_trs
        end = ((n_segments - seg_idx) + 1) * seg_trs
        if method == 'wmb':
            segment_isc.append(isc_func(data[0][start:end], data[1][start:end]))
        else:
            segment_isc.append(isc_func(data[start : end]))
        seg_idx -= 1
    return np.array(segment_isc)


def tr_mask_from_segments(n_trs, seg_mask):
    """Filter out TRs of timecourse data using a mask representing TR segments"""
    assert n_trs % len(seg_mask) == 0, f"n_trs {n_trs} is not divisible into {len(seg_mask)} segments"
    
    # infer how many TRs are represented by the seg_mask
    seg_size_trs = n_trs // len(seg_mask)
    tr_mask = np.full(n_trs, False)
    for i, seg in enumerate(seg_mask):
        if seg == True:
            tr_mask[seg_size_trs*i : seg_size_trs*(i+1)] = np.full(seg_size_trs, True)
    return tr_mask


def filter_segment_trs(data, seg_mask, axis=0):
    """Get TR subset of original data based on seg_mask"""
    assert data.shape[axis] % len(seg_mask) == 0, f"data size on axis {axis} ({data.shape[axis]}) is not divisible into {len(seg_mask)} segments"
    tr_mask = tr_mask_from_segments(data.shape[axis], seg_mask)
    tr_indices = np.where(tr_mask==True)
    return np.squeeze(np.take(data, tr_indices, axis=axis))


def movie_seg_compute(seg_mask, func, *data):
    """
    Obtain a subset of TRs using a TR segment mask, then compute a function
    on the new timeseries data.

    Axis=0 of the given data array is assumed to represent TRs by default.
    More than one data array can be provided.

    EG:
    stimulus_present_isc = movie_seg_compute([True, False, True, False],
                                    partial(isc_func, n_jobs=-1),
                                    movie_data)
    """
    data_subset = [filter_segment_trs(d, seg_mask, axis=0) for d in data]
    return func(*data_subset)
    

def timestamps_from_segments(n_segs, n_trs, TR, return_as="h:m:s"):
    """Calculate the timestamps that correspond to n_segs out of n_trs for some 
    timecourse data.
    
    TR is assumed to be given in seconds
    """
    assert n_trs % n_segs == 0, f"n_trs ({n_trs}) must be divisible by seg_mask size ({n_segs})"
    seg_size_trs = n_trs // n_segs
    seg_size_seconds = seg_size_trs * TR 
    timestamps = []
    for seg in range(n_segs):
        timestamps.append(timedelta(seconds=seg_size_seconds*(seg+1)))
        
    if return_as == "h:m:s":
        return [f"{t}" for t in timestamps]
    elif return_as == "seconds":
        return [t.seconds for t in timestamps] 


# =======================
# RSA-based ISC functions
# =======================


def finn_isrsa(pwise_isc=None, pwise_behav=None, 
               neural_data=None, behav_data=None,
               pwise_func=None, tri_func=None, 
               n_jobs=None, joblib_kwargs={}):   
    """Calculate intersubject representational similarity analysis (IS-RSA) as
    described in Finn et al. (2020) [1].
    
    
    References:
    [1]
    Finn, E. S., Glerean, E., Khojandi, A. Y., Nielson, D., Molfese, P. J., 
    Handwerker, D. A., & Bandettini, P. A. (2020). Idiosynchrony: From shared 
    responses to individual differences during naturalistic neuroimaging. 
    NeuroImage, 215, 116828.
    """
    assert pwise_isc.ndim == 2, f"pwise_isc ndim must be 2; you provided {pwise_isc.ndim}"
    assert pwise_isc.shape[1] == len(pwise_behav), f"pwise_isc.shape[1] and len(pwise_behav) must be equal, but were {pwise_isc.shape} and {len(pwise_behav)} instead"
    assert callable(tri_func) or tri_func in [None, 'spearman', 'pearson'], f"tri_func must either be a callable object, spearman, pearson, or None (which defaults to 'spearman')"
    assert isinstance(n_jobs, (type(None), int)), f"n_jobs was type '{n_jobs}', but must be NoneType or int"
    assert not (neural_data is None and pwise_isc is None), f"Either neural_data or pwise_isc must be provided; both cannot be None"
    assert not (behav_data is None and pwise_behav is None), f"Either behav_data or pwise_behave must be provided; both cannot be None"
    if pwise_behav is None:
        assert callable(pwise_func) or type(pwise_func)==str, f"pwise_func must either be a callable object (ie., function, method) or a str; it cannot be None"
    # if pwise_isc is not None and pwise_behav is not None:
    #     if  

    logger.debug(f"Running finn_isrsa()")

    if tri_func in ['spearman', None]:
        tri_func = lambda x1, x2: spearmanr(x1, x2)[0]
    elif tri_func == 'pearson':
        tri_func = lambda x1, x2: pearsonr(x1, x2)[0]
    
    
    if pwise_isc is None:
        pwise_isc = isc(neural_data, pairwise=True, n_jobs=n_jobs)
    if pwise_behav is None:
        pwise_behav = pwise_func(behav_data)
    
    if n_jobs is not None:
        # pwise_behav = tri2vect(pwise_behav)
        with Parallel(n_jobs, **joblib_kwargs) as parallel:
            isrsa_by_node = parallel(delayed(tri_func)(pwise_isc[i], pwise_behav)
                                        for i in range(pwise_isc.shape[0]))
        
    else:
        # pwise_behav = tri2vect(pwise_behav)
        isrsa_by_node = []
        for i in range(pwise_isc.shape[0]): # assumes pwise_isc input is (n_pairs, n_nodes)
            isrsa_by_node.append(tri_func(
                        pwise_isc[i], 
                        pwise_behav))
    
    return np.array(isrsa_by_node)

# Define helper functions from finn is-rsa tutorial
def reorder_square_mtx(mtx, sort_idx, sort=True):
    """Sorts rows/columns of a matrix according to a separate vector."""
    inds = sort_idx.argsort()
    mtx_sorted = mtx.copy()
    if type(mtx_sorted) == pd.DataFrame:
        mtx_sorted = mtx_sorted.to_numpy()
    mtx_sorted = mtx_sorted[inds, :]
    mtx_sorted = mtx_sorted[:, inds]
    return mtx_sorted
    
def scale_mtx(mtx):
    return (mtx-np.min(mtx)) / (np.max(mtx) - np.min(mtx))

def tri2vect(mtx, upper=True, sort_idx=None, k=None):
    """Get triangle of a square matrix as a vector"""
    if type(mtx) == pd.DataFrame:
        mtx = mtx.to_numpy()
    if sort_idx is not None:
        mtx = reorder_square_mtx(mtx, sort_idx, sort=True)
        
    if k is None:
        if upper:
            k = 1
        else:
            k = -1
    if upper:
        trifunc = partial(np.triu_indices, k=k)
    else:
        trifunc = partial(np.tril_indices, k=k)
    return mtx[trifunc(mtx.shape[0])]

def pair_scalar(fn, x):
    out = np.full((x.shape[0], x.shape[0]), np.nan)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            out[i, j] = fn(x[i], x[j])
    return out

def avg_corr(avg_fn, data, above_zero=True):
    """
    Return the average positive (or negative) correlation in an array
    of correlation values range from -1 to 1. 
    """
    if above_zero:
        data = np.where(data > 0.0, data, np.nan)
    else: 
        data = np.where(data < 0.0, data, np.nan)
    return avg_fn(data)
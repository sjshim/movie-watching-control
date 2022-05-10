# intersubject.py: Module contains an Intersubject class that provides
# methods for computing one-sample and within-between 
# intersubject correlation and intersubject functional correlation

from functools import partial

import numpy as np

from joblib import Parallel, delayed
from nibabel import Nifti1Image
from brainiak.utils.utils import array_correlation, _check_timeseries_input
from brainiak.isc import _threshold_nans

from .tools import save_data
from .basic_stats import compute_r, r_average

def isc(data, pairwise=False, summary_statistic=None, tolerate_nans=True, n_jobs=None):
    """Brainiak ISC implemntation but with Joblib parallelisation."""

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
        if n_jobs in (1, None):
            voxel_iscs = Parallel(n_jobs=n_jobs)\
                            (delayed(pairwise_corr)(data[:,v,:])
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
        if n_jobs not in (1, None):
            iscs_stack = Parallel(n_jobs=n_jobs)\
                            (delayed(loo_corr)(data, s)
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

    return iscs

def wmb_isc(d1, d2, subtract_wmb=False, tolerate_nans=True, n_jobs=None, avg_kind=None):
    
    assert d1.shape == d2.shape, "d1 and d2 must have equal shapes"
    d1, d1_n_TRs, d1_n_voxels, d1_n_subs = _check_timeseries_input(d1)
    d2, d2_n_TRs, d2_n_voxels, d2_n_subs = _check_timeseries_input(d2)
    
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    d1, d1_mask = _threshold_nans(d1, tolerate_nans)
    d2, d2_mask = _threshold_nans(d2, tolerate_nans)
    
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
    for idx, d in enumerate(data_tup):
        if n_jobs not in (1, None):
            w_iscs_stack += Parallel(n_jobs=n_jobs)\
                             (delayed(loo_corr)(data_tup[idx], s)
                             for s in range(d1_n_subs))
            
            b_iscs_stack += Parallel(n_jobs=n_jobs)\
                              (delayed(one2avg_corr)(data_tup[idx][...,s], data_tup[idx-1])
                              for s in range(d1_n_subs))
            
        else:
            for s in range(d1_n_subs):
                w_iscs_stack.append(loo_corr(data_tup[idx], s))
                
                b_iscs_stack.append(one2avg_corr(data_tup[idx][...,s], 
                                                data_tup[idx-1]))
    
    w_iscs_stack, b_iscs_stack = np.array(w_iscs_stack), np.array(b_iscs_stack)
    
    # Get original data shape after masking out NaNs
    within_isc = np.full((w_iscs_stack.shape[0], d1_n_voxels), np.nan)
    between_isc = np.full((b_iscs_stack.shape[0], d1_n_voxels), np.nan)
    within_isc[:, np.where(d1_mask)[0]] = w_iscs_stack
    between_isc[:, np.where(d1_mask)[0]] = b_iscs_stack
    
    if subtract_wmb:
        wmb = within_isc - between_isc
        if avg_kind:
            return avg_kind(wmb, axis=0)
        else:
            return wmb
    else:
        return np.array([within_isc, between_isc])

def segment_isc(data, seg_trs, method='loo', summary_statistic=None, tolerate_nans=True, 
                subtract_wmb=False, n_jobs=None):
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

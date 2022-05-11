# nonparametric.py: This module contains functions for nonparametric assessment
# and thresholding of fMRI data produced during intersubject analysis.

import os
from functools import partial
import numpy as np
from scipy.stats import (pearsonr, spearmanr, rankdata, ttest_ind,
ttest_rel, ttest_1samp)
from joblib import Parallel, delayed, dump, load


def null_threshold(observed, null_dist, alpha=0.05, max_stat=False):
    """
    Threshold observed statistics using a null distribution of the statistic.
    Thresholding can optionally be performed using the maximum statistic
    from each iteration of the provided null distribution.
    """
    # Obtain max stat distribution then threshold observed data 
    if max_stat:
        null_dist = np.nanmax(null_dist, axis=1)
        null_count = []
        for i in range(observed.shape[0]):
            null_count.append((null_dist > observed[i]).sum())
        null_count = np.array(null_count)
#         mask = (null_count / observed.shape[0]) < alpha

    # Directly threshold observed data using provided null distribution
    else:
        null_count = (null_dist > observed).sum(axis=0)
    mask = (null_count / null_dist.shape[0]) < alpha   
    
    return np.where(mask==True, observed, np.nan)


# sign flip permutation test
def perm_signflip(x, stat_func=partial(np.mean, axis=0), 
                  n_iter=100,
                  tail='greater', 
                  apply_threshold=False, 
                  threshold_kwargs={'alpha':0.05, 'max_stat':False},
                  n_jobs=None,
                  joblib_kwargs={}):
    """
    Perform a signflip permutation test on observed statistical results
    The positive/negative sign for the data from a random subset of subjects,
    then a function is computed on the randomized data; this is performed n_iter
    times and the null distribution of statistics is returned.
    
    x must be an array with shape (n_features x n_samples)
    """
#     if avg_kind == 'mean':
#         avg = partial(np.mean, axis=0)
#     elif avg_kind == 'median':
#         avg = partial(np.median, axis=0)
    
    def flip_and_compute(x, seed=None):
#         if seed:
# #             np.random.seed(seed)
        this_rng = np.random.default_rng(seed)
#         print(seed)
#         print(this_rng)
        sign_flip = this_rng.choice([-1, 1], size=(x.shape[0]))[:,np.newaxis]
        return stat_func(x*sign_flip)
    
    if n_jobs not in (1, None):
        null_dist = Parallel(n_jobs=n_jobs, **joblib_kwargs)\
                        (delayed(flip_and_compute)(x, seed=i)
                        for i in range(n_iter))
    else:
        null_dist = []
        for i in range(n_iter):
            null_dist.append(flip_and_compute(x, seed=i))
            
    null_dist = np.array(null_dist)
    if apply_threshold:
        return null_threshold(stat_func(x), null_dist, **threshold_kwargs)
    else:
        return null_dist


def perm_grouplabel(x1, x2, stat_func, 
                    n_iter=100, 
                    tail='greater', 
                    avg_kind='mean', 
                    apply_threshold=False, 
                    threshold_kwargs={'alpha':0.05, 'max_stat':False}, 
                    n_jobs=None, 
                    memmap_dir=None,
                    joblib_kwargs={}):
    """
    Perform group label permutation test. 
    
    Two groups of subject level data are shuffled, then the provided 
    function is computed on the new groups; this is performed n_iter times
    and the null distribution of statistics is returned.
    """
#     if avg_kind == 'mean':
#         avg = np.mean
#     elif avg_kind == 'median':
#         avg = np.median

    def shuffle_and_compute(a, b, seed=None):
        print(f"iteration {seed}")
#         print(f"a shape: {a.shape}\nb shape: {b.shape}")
#         print(f"a len: {len(a)}")
#         if seed:
#             np.random.seed(seed)
        this_rng = np.random.default_rng(seed)
        c = np.append(a.T, b.T, axis=0) # temporary fix for the transpose issue from my isc functions 
#         print(f"c shape before shuffle: {c.shape}")
        this_rng.shuffle(c)
#         print(f"c shape after shuffle: {c.shape}")
        b = c[len(a.T):] # same fix
        a = c[:len(a.T)]
        del c
#         print(f"c shape: {c.shape}\nd shape: {d.shape}")
        return stat_func(a.T, b.T)
    
    if n_jobs not in (1, None):
        if memmap_dir:
            x1_dir = os.path.join(memmap_dir, 'group1')
            x2_dir = os.path.join(memmap_dir, 'group2')
            dump(x1, x1_dir)
            dump(x2, x2_dir)
            x1 = load(x1_dir, mmap_mode='r')
            x2 = load(x2_dir, mmap_mode='r')
            
        null_dist = Parallel(n_jobs=n_jobs, **joblib_kwargs)\
                        (delayed(shuffle_and_compute)(x1, x2, i)
                        for i in range(n_iter))
    else:
        null_dist = []
        for i in range(n_iter):
            null_dist.append(shuffle_and_compute(x1, x2, i))
    
#     null_dist = avg(null_dist, axis=1)
    null_dist = np.array(null_dist)
#     print(f"null dist shape: {null_dist.shape}")
    if apply_threshold:
#         return null_threshold(avg(stat_func(x1, x2), axis=0), null_dist, **threshold_kwargs)
        return null_threshold(stat_func(x1, x2), null_dist, **threshold_kwargs)
    else:
        return null_dist


def perm_mantel(x1, x2, tri_func='spearman', 
                n_iter=100, 
                apply_threshold=False,
                threshold_kw={'alpha':0.05, 'max_stat':False},
                n_jobs=None, 
                joblib_kw={}):

    """
    Perform mantel permutation test to assess the significance of correlation
    between two pairwise matrices' upper triangles. The test is performed by
    randomly shuffling the rows and columns of one of the two matrices, then
    correlation the two triangles over n iterations.
    """
    
    if tri_func == 'spearman':
        tri_func = spearmanr
    elif tri_func == 'pearson':
        tri_func = pearsonr
    elif tri_func is None:
        tri_func = spearmanr
    
    def shuffle_and_tri_func(a, b, seed=None):
        rng = np.random.default_rng(seed)
        rng.shuffle(b)
        return tri_func(a, b)
    
    if n_jobs in (1, None):
        null_dist = Parallel(n_jobs=n_jobs, **joblib_kw)\
                    (delayed(shuffle_and_tri_func)(x1, x2, seed=i)
                    for i in range(n_iter))
    else:
        null_dist = []
        for i in range(n_iter):
            null_dist.append(shuffle_and_tri_func(x1, x2, seed=i))
    null_dist = np.array(null_dist)
    
    if apply_threshold:
        return null_threshold(tri_func(x1, x2), null_dist, **threshold_kw)
    else:
        return null_dist
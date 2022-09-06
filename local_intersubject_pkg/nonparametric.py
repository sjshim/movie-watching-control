# nonparametric.py: This module contains functions for nonparametric assessment
# and thresholding of fMRI data produced during intersubject analysis.

import os
from pathlib import Path
from functools import partial
import logging

import numpy as np
from scipy.stats import (pearsonr, spearmanr, rankdata, ttest_ind,
ttest_rel, ttest_1samp)
from joblib import Parallel, delayed, dump, load

logger = logging.getLogger(__name__)

valid_tail_args = ['upper', 'lower', 'both']
valid_return_type_args = ['sig_vals', 'null_mask', 'null_ct', 'pvals', 'stat_mask'] 
valid_null_threshold_kwargs = ['alpha', 'tail', 'max_stat', 'return_type']

def null_threshold(observed : np.ndarray, 
                    null_dist : np.ndarray, 
                    alpha: float = 0.05, 
                    tail: str = 'upper', 
                    max_stat: bool = False, 
                    return_type: str = 'sig_vals') -> np.ndarray: 
    """
    Threshold observed statistics using a null distribution of the statistic.
    Thresholding can optionally be performed using the maximum statistic
    from each iteration of the provided null distribution.

    The steps of this function involve the following:
        - check whether the observed statistics were less extreme than the null
          statistic; return True if null is more extreme and False if observed is
          more extreme
        - count the number of True values, then divide the number of True values
          by the total number of null values
        - compare to see that the True proportion is less than alpha;
          if True, then that observed statistic is judged to be significant and
          is returned, if False then return np.NaN
    """
    assert isinstance(observed, np.ndarray), print(f"observed was type '{type(observed)}', but must be np.ndarray")
    assert isinstance(null_dist, np.ndarray), print(f"null_dist was type '{type(null_dist)}', but must be np.ndarray")
    assert observed.shape[0] in null_dist.shape, print(f"observed.shape[0] and null_dist.shape[1] must be the same, but were '{observed.shape}' and '{null_dist.shape}' instead")
    assert null_dist.ndim <= 2, print(f"null_dist had {null_dist.ndim} dimensions, but must be 2 or less")
    assert isinstance(alpha, float), print(f"alpha was type '{type(alpha)}', but must be float")
    assert tail in valid_tail_args, print(f"tail was '{tail}', but should be 'upper', 'lower', or 'both'")
    assert max_stat in [True, False], print(f"max_stat was '{max_stat}', but must be either True or False")
    assert return_type in valid_return_type_args, print(f"return_type was '{return_type}', but must be 'sig_vals', 'null_mask', 'stat_mask', 'null_ct', or 'pvals")
    assert not (max_stat == True and return_type == 'null_mask'), print(f"max_stat=True cannot be used with return_type='null_mask'")
    logger.debug(f"Running null_threshold()")

    if null_dist.ndim == 1:
        null_dist = null_dist[None, :]
    compare_func = {
        'upper': (lambda d, n: d > n),
        'lower': (lambda d, n: d < n),
        'both': (lambda d, n: d>n or d<n)
    }
    operator = compare_func[tail]

    try:
        # Obtain max stat distribution then threshold observed data 
        if max_stat:
            null_dist = np.nanmax(null_dist, axis=1)
            out = np.array([operator(null_dist, observed[i]).sum() for i in range(observed.shape[0])])

        # Directly threshold observed data using provided null distribution
        else:
            out = operator(null_dist, observed)
            if return_type == 'null_mask':
                logger.debug(f"return_type=={return_type}\n{out}")
                return out
            out = out.sum(axis=0)

        if return_type == 'null_ct':
            logger.debug(f"return_type=={return_type}\n{out}")
            return out
        out = out / null_dist.shape[0]
        if return_type == 'pvals':
            logger.debug(f"return_type=={return_type}\n{out}")
            return out
        out = out < alpha
        if return_type == 'stat_mask':
            logger.debug(f"return_type=={return_type}\n{out}")
            return out
        out = np.where(out==True, observed, np.nan)
        logger.debug(f"return_type=={return_type}\n{out}")
        return out

    except BaseException as err:
        logger.exception(err)
        raise


# sign flip permutation test
def perm_signflip(x: np.ndarray, 
                stat_func=partial(np.mean, axis=0), 
                n_iter: int = 100,
                tail: str = 'upper', 
                apply_threshold: bool = False, 
                threshold_kwargs: dict = {'alpha':0.05, 'max_stat':False},
                n_jobs=None,
                seed=None,
                joblib_kwargs={}) -> np.ndarray:
    """
    Perform a signflip permutation test on observed statistical results
    The positive/negative sign for the data from a random subset of subjects,
    then a function is computed on the randomized data; this is performed n_iter
    times and the null distribution of statistics is returned.
    
    x must be an array with shape (n_features x n_samples)
    """
    assert isinstance(x, np.ndarray), print(f"x was type '{type(x)}', but should be np.ndarray")
    assert x.ndim == 2 and x.shape[0] > 1, print(f"x was shape {x.shape}, but must have 2 dimensions and x.shape[0] (n_features) > 1")
    assert isinstance(stat_func, (type(lambda x:''), type(partial(abs)))), f"stat_func was type '{type(stat_func)}', but should be a function object or {type(partial(abs))}"
    assert tail in valid_tail_args, print(f"tail was '{tail}', but must be {valid_tail_args}")
    assert type(apply_threshold)==bool, print(f"apply_threshold was type '{type(apply_threshold)}', but must be bool")
    assert set(threshold_kwargs).issubset(valid_null_threshold_kwargs), \
        f"threshold_kwargs keys were {list(threshold_kwargs)}, but they must contain some or all of {valid_null_threshold_kwargs}"
    assert isinstance(n_jobs, (type(None), int)), f"n_jobs was type '{type(n_jobs)}', but must be None or an int"
    assert isinstance(joblib_kwargs, (type(None), dict)), print(f"joblib_kwargs was type '{type(joblib_kwargs)}', but must be None or a dict")

    rng = np.random.default_rng(seed=seed)
    def flip_and_compute(x, seed=None):
        this_rng = np.random.default_rng(seed)
        sign_flip = this_rng.choice([-1, 1], size=(x.shape[0]))[:,np.newaxis]
        return stat_func(x*sign_flip)
    
    if n_jobs not in (1, None):
        with Parallel(n_jobs=n_jobs, **joblib_kwargs) as parallel:
            null_dist = parallel(delayed(flip_and_compute)(x, seed=rng.integers(0,n_iter))
                                for _ in range(n_iter))
    else:
        null_dist = []
        for _ in range(n_iter):
            null_dist.append(flip_and_compute(x, seed=rng.integers(0,n_iter)))
            
    null_dist = np.array(null_dist)
    if apply_threshold:
        return null_threshold(stat_func(x), null_dist, **threshold_kwargs)
    else:
        return null_dist


def perm_grouplabel(x1, x2, stat_func, 
                    n_iter=100, 
                    tail='upper', 
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
    assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray), \
        print(f"Both x1 and x2 must be type np.ndarray, but instead were '{type(x1)}' and '{type(x2)}'")
    assert isinstance(stat_func, type(lambda x:'')), print(f"stat_func was type '{type(stat_func)}', but must be a function object")
    assert tail in valid_tail_args, f"tail was '{tail}', but must be {valid_tail_args}"
    assert avg_kind in ['mean', 'median'], print(f"avg_kind must be 'mean' or 'median'")
    assert type(apply_threshold) == bool, print(f"apply_threshold was type '{type(apply_threshold)}', but must be bool")
    assert set(threshold_kwargs).issubset(valid_null_threshold_kwargs), \
            f"threshold_kwargs keys were {list(threshold_kwargs)}, but they must contain some or all of {valid_null_threshold_kwargs}"
    assert isinstance(n_jobs, (type(None), int)), print(f"n_jobs was type '{type(n_jobs)}', but must be None or an int")
    assert isinstance(memmap_dir, (type(None), str, Path)), print(f"memmap_dir was type '{type(memmap_dir)}', but must either be None or path-like (str or pathlib.Path)")
    assert isinstance(joblib_kwargs, (type(None), dict)), print(f"joblib_kwargs was type '{type(joblib_kwargs)}', but must be None or a dict")
    

    # if tail not in valid_tail_args: raise ValueError(f"tail was '{tail}', but must be {valid_tail_args}")
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
                threshold_kwargs={'alpha':0.05, 'max_stat':False},
                n_jobs=None, 
                joblib_kwargs={}):

    """
    Perform mantel permutation test to assess the significance of correlation
    between two pairwise matrices' upper triangles. The test is performed by
    randomly shuffling the rows and columns of one of the two matrices, then
    correlation the two triangles over n iterations.
    """
    assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray), \
        print(f"Both x1 and x2 must be type np.ndarray, but instead were '{type(x1)}' and '{type(x2)}'")
    assert type(tri_func)==type(lambda _:'') or tri_func in ['spearman', 'pearson', None], \
        print(f"tri_func was '{tri_func}', but must either be a function object, or 'spearman', 'pearson', or None")
    assert type(apply_threshold) == bool, print(f"apply_threshold was type '{type(apply_threshold)}', but must be bool")
    assert set(threshold_kwargs).issubset(valid_null_threshold_kwargs), \
            f"threshold_kwargs keys were {list(threshold_kwargs)}, but they must contain some or all of {valid_null_threshold_kwargs}"
    assert isinstance(n_jobs, (type(None), int)), print(f"n_jobs was type '{type(n_jobs)}', but must be None or an int")
    assert isinstance(joblib_kwargs, (type(None), dict)), print(f"joblib_kwargs was type '{type(joblib_kwargs)}', but must be None or a dict")

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
        null_dist = Parallel(n_jobs=n_jobs, **joblib_kwargs)\
                    (delayed(shuffle_and_tri_func)(x1, x2, seed=i)
                    for i in range(n_iter))
    else:
        null_dist = []
        for i in range(n_iter):
            null_dist.append(shuffle_and_tri_func(x1, x2, seed=i))
    null_dist = np.array(null_dist)
    
    if apply_threshold:
        return null_threshold(tri_func(x1, x2), null_dist, **threshold_kwargs)
    else:
        return null_dist
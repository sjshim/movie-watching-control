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

    Parameters
    ----------
    observed : np.ndarray (1-dimensional)
        The observed stats that will be thresholded using null_dist

    null_dist : np.ndarray
        The null distribution used to thresold observed with.

        If null_dist.ndim==1, then the length of null_dist should be equal 
        to that of observed. Otherwise, null_dist.shape == (n_iterations, n_stats),
        and null_dist.shape[1] should be equal to the length of observed.

    alpha : float, default=0.05
        The significance level used to threshold 'observed' with.

        An observed value is thresholded as significant if the number of
        null values (from null_dist) that are more extreme than observed[i]
        is less than alpha.

    tail : str, default='upper'
        Choose whether to test which distribution tails to observed for 
        significance with.

        Options: 'upper', 'lower', 'both'

    max_stat : bool, default=False
        If max_stat=False, values of observed are tested for significance
        by comparing them only to the same index across all iterations of
        null_dist.

        If max_stat=True, the max stats of null_dist across n_stats (axis=1) is
        collected, then each observed value is tested for significance
        across n_iterations (axis=0) max values.

    return_type : str, default='sig_vals'
        Chooose whether to return an intermediate step of the thresholded process.

        Options:
            - 'sig_vals'
              Return the final thresholded data from 'observed';
              significant values are saved as-is, and non-significant values
              are replaced with np.nan. Shape=(n_stats,)

            - 'null_mask': (only works with max_stat=False)
              Return the mask representing elements in 'null_dist' where 
              null values are more extreme than 'observed'. Not currently available
              for max_stat=True. For the full null dist, shape=(n_iteratons, n_stats)

            - 'null_ct':
              Return the count of times where null values were more extreme
              than each observed value. Shape=(n_stats,)

            - 'pvals':
              Return pvals for each observed value (pvals = null_ct / n_iterations).
              Shape=(n_stats,). 

            - 'stat_mask':
              Return the final mask used to finally threshold 'observed' with.
              This is obtained by finding values where stat_mask < alpha.
              Shape=(n_stats,)

    Returns
    -------
    Return array and dimensions based on return_type kwarg.

    """
    assert isinstance(observed, np.ndarray), print(f"observed was type '{type(observed)}', but must be np.ndarray")
    assert isinstance(null_dist, np.ndarray), print(f"null_dist was type '{type(null_dist)}', but must be np.ndarray")
    assert observed.ndim == 1 and null_dist.ndim in [1,2], \
        print(f"Number of dimensions for observed should be 1 and for null_dist should be 1 or 2, but instead were {observed.ndim} and {null_dist.ndim}, respectively")
    if null_dist.ndim == 1:
        assert len(observed) == len(null_dist), f"Both observed and null_dist should be same length, but instead were {observed.shape} and {null_dist.shape}"
    else:
        assert len(observed) == null_dist.shape[1], f"Length of observed should be the same as null_dist.shape[1], but instead were {observed.shape} and {null_dist.shape}"
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
            if tail=='upper':
                null_dist = np.nanmax(null_dist, axis=1)
            elif tail=='lower':
                null_dist = np.nanmin(null_dist, axis=1)
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
                stat_func=partial(np.mean, axis=1), 
                n_iter: int = 100, 
                apply_threshold: bool = False,
                tail: str = 'upper', 
                threshold_kwargs: dict = {'alpha':0.05, 'max_stat':False},
                n_jobs: int = None,
                seed: int = None,
                joblib_kwargs: dict = {}) -> np.ndarray:
    """
    Perform a signflip permutation test on observed statistical results
    The positive/negative sign for the data from a random subset of subjects,
    then a function is computed on the randomized data; this is performed n_iter
    times and the null distribution of statistics is returned.
    
    x must be an array with shape (n_samples, n_features)

    Parameters
    ---------
    x : np.ndarray
        The n_samples x n_features input data to apply sign-flipping onto in order
        to generate a null distribution.

    stat_func : callable object, default=partial(np.mean, axis=1)
        The function that is applied to x after sign-flipping; this function's
        results are stored in the final null distribution list.

        By default, x.mean(axis=1) is called to get the average n_samples
        across all features.

    n_iter : int, default=100
        The number of iterations to perform sign flipping permutations for.

        n_iter is 100 by default, but at least 1000 is probably sufficient.

    Returns
    -------
    null distribution, shape=(n_iterations, n_sample_stats)
    
    """
    assert isinstance(x, np.ndarray), print(f"x was type '{type(x)}', but should be np.ndarray")
    assert x.ndim == 2 and x.shape[1] > 1, print(f"x was shape {x.shape}, but must have 2 dimensions and x.shape[1] (n_features) > 1")
    assert callable(stat_func), f"stat_func {stat_func} was type '{type(stat_func)}', but should be a callable object (eg., function or method)"
    assert tail in valid_tail_args, f"tail was '{tail}', but must be {valid_tail_args}"
    assert type(apply_threshold)==bool, print(f"apply_threshold was type '{type(apply_threshold)}', but must be bool")
    assert set(threshold_kwargs).issubset(valid_null_threshold_kwargs), \
        f"threshold_kwargs keys were {list(threshold_kwargs)}, but they must contain some or all of {valid_null_threshold_kwargs}"
    assert isinstance(n_jobs, (type(None), int)), f"n_jobs was type '{type(n_jobs)}', but must be None or an int"
    assert isinstance(joblib_kwargs, (type(None), dict)), print(f"joblib_kwargs was type '{type(joblib_kwargs)}', but must be None or a dict")
    
    logger.debug(f"Running perm_signflip(tail={tail}, apply_threshold={apply_threshold}, n_jobs={n_jobs}, seed={seed})")
    # logger.debug(f"Using top level rng seed={seed} for {n_iter} iterations")
    
    rng = np.random.default_rng(seed=seed)
    def flip_and_compute(this_seed=None):
        this_rng = np.random.default_rng(this_seed)
        # flip all stats for a random subset of features
        sign_flip = this_rng.choice([-1, 1], size=(x.shape[1]))
        return stat_func(x*sign_flip)
    
    if n_jobs is not None:
        with Parallel(n_jobs=n_jobs, **joblib_kwargs) as parallel:
            null_dist = parallel(delayed(flip_and_compute)(this_seed=rng.integers(0,n_iter))
                                for _ in range(n_iter))
    else:
        null_dist = []
        for _ in range(n_iter):
            null_dist.append(flip_and_compute(this_seed=rng.integers(0,n_iter)))
            
    null_dist = np.array(null_dist)
    if apply_threshold:
        return null_threshold(stat_func(x), null_dist, 
                                tail=tail, **threshold_kwargs)
    else:
        return null_dist


def perm_grouplabel(x1: np.ndarray, 
                    x2: np.ndarray, 
                    stat_func, 
                    n_iter: int = 100,  
                    avg_kind: str = 'mean', 
                    apply_threshold: bool = False, 
                    tail: str = 'upper',
                    threshold_kwargs: dict = {'alpha':0.05, 'max_stat':False}, 
                    n_jobs: int = None, 
                    seed: int = None,
                    # memmap_dir=None,
                    joblib_kwargs: dict = {}) -> np.ndarray:
    """
    Perform group label permutation test. 
    
    Two groups of subject level data are shuffled, then the provided 
    function is computed on the new groups; this is performed n_iter times
    and the null distribution of statistics is returned.

    x1 and x2 dimension should represent (n_samples, n_features)

    Parameters
    ----------
    x1, x2 : np.ndarray
        Two groups of n_samples x n_features input data to apply group label
        shuffling onto to generate a null distribution.

        x1 and x2 n_samples must be equal, but can have different n_features.

    stat_func : callable object
        A function that is applied onto x1 and x2 after group label shuffling;
        this function's results are stored in the final null distribution list.

    n_iter : int, default=100
        The numbger of iterations to perform group label shuffling permutations
        for.

        n_iter is 100 by default, but at least 1000 is probably sufficient.

    Returns
    -------
    null distribution, shape=(n_iterations, n_sample_stats)
    """
    assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray), \
        print(f"Both x1 and x2 must be type np.ndarray, but instead were '{type(x1)}' and '{type(x2)}'")
    assert x1.shape[0] == x2.shape[0], f"x1.shape[0] and x2.shape[0] must be equal, but were {x1.shape} and {x2.shape} instead"
    assert callable(stat_func), f"stat_func {stat_func} was type '{type(stat_func)}', but should be a callable object (eg., function or method)"
    assert tail in valid_tail_args, f"tail was '{tail}', but must be {valid_tail_args}"
    assert avg_kind in ['mean', 'median'], print(f"avg_kind must be 'mean' or 'median'")
    assert type(apply_threshold) == bool, print(f"apply_threshold was type '{type(apply_threshold)}', but must be bool")
    assert set(threshold_kwargs).issubset(valid_null_threshold_kwargs), \
            f"threshold_kwargs keys were {list(threshold_kwargs)}, but they must contain some or all of {valid_null_threshold_kwargs}"
    assert isinstance(n_jobs, (type(None), int)), print(f"n_jobs was type '{type(n_jobs)}', but must be None or an int")
    # assert isinstance(memmap_dir, (type(None), str, Path)), print(f"memmap_dir was type '{type(memmap_dir)}', but must either be None or path-like (str or pathlib.Path)")
    assert isinstance(joblib_kwargs, (type(None), dict)), print(f"joblib_kwargs was type '{type(joblib_kwargs)}', but must be None or a dict")
    
    rng = np.random.default_rng(seed=seed)
    def shuffle_and_compute(a, b, this_seed=None):
        this_rng = np.random.default_rng(this_seed)
        c = np.append(a, b, axis=1) # temporary fix for the transpose issue from my isc functions 
        this_rng.shuffle(c, axis=1)
        b = c[:, a.shape[1]: ] # same fix
        a = c[:, : a.shape[1]]
        del c
        return stat_func(a, b)
    
    if n_jobs is not None:
        with Parallel(n_jobs=n_jobs, **joblib_kwargs) as parallel:
            null_dist = parallel(delayed(shuffle_and_compute)(x1, x2, this_seed=rng.integers(0,n_iter))
                                for _ in range(n_iter))
    else:
        null_dist = []
        for _ in range(n_iter):
            null_dist.append(shuffle_and_compute(x1, x2, this_seed=rng.integers(0,n_iter)))
    
    null_dist = np.array(null_dist)
    if apply_threshold:
        return null_threshold(stat_func(x1, x2), null_dist, 
                                tail=tail, **threshold_kwargs)
    else:
        return null_dist


def perm_mantel(x_n, x_b, tri_func='spearman', 
                n_iter=100, 
                apply_threshold=False,
                tail='upper',
                threshold_kwargs={'alpha':0.05, 'max_stat':False},
                seed = None,
                n_jobs=None, 
                joblib_kwargs={}):

    """
    Perform mantel permutation test to assess the significance of correlation
    between two pairwise matrices' upper triangles. The test is performed by
    randomly shuffling the rows and columns of one of the two matrices, then
    correlation the two triangles over n iterations.

    Parameters
    ----------
    x_n : np.ndarray
        Subject-pairwise ISC data. Shape=(n_regions, n_subject_pairs)

    x_b : np.ndarray
        Subject-pairwise behavioral data. Shape=(n_subject_pairs,)

    tri_func : callable, str, or None, default='spearman'
        The method to compare subject pairs of x_n and x_b. 

    apply_threshold : bool, default=True
        Choose to apply null_threshold() to data. If false, then
        null distribution is returned as-is.

    tail : str, default='upper'
        Choose which side of the null distribution is tested if
        apply_threshold=True. 

        Options: 'upper', 'lower', or 'both'

    Returns
    -------
    null distribution, shape=(n_iterations, n_sample_stats)
    """
    assert isinstance(x_n, np.ndarray) and isinstance(x_b, np.ndarray), \
        print(f"Both x_n and x_b must be type np.ndarray, but instead were '{type(x_n)}' and '{type(x_n)}'")
    assert x_n.ndim<=2, f"x_n neural pwise data should be <= 2 dimensions, but was {x_n.ndim} instead"
    assert x_b.ndim==1, f"x_b behavioral pwise data should be 1-d, but was {x_b.ndim} instead"
    assert x_n.shape[1] == x_b.shape[0], f"x_n.shape[1] and x_b.shape[0] n_subject_pairs dimension should be equal, but were {x_n.shape} and {x_b.shape} instead"
    assert callable(tri_func) or tri_func in ['spearman', 'pearson', None], \
        print(f"tri_func was '{tri_func}', but must either be a callabe object, 'spearman', 'pearson', or None (which defaults to 'spearman'")
    assert type(n_iter)==int, f"n_iter was type '{type(n_iter)}', but should be int"
    assert tail in valid_tail_args, f"tail was '{tail}', but must be be {valid_tail_args}"
    assert type(apply_threshold) == bool, print(f"apply_threshold was type '{type(apply_threshold)}', but must be bool")
    assert set(threshold_kwargs).issubset(valid_null_threshold_kwargs), \
            f"threshold_kwargs keys were {list(threshold_kwargs)}, but they must contain some or all of {valid_null_threshold_kwargs}"
    assert isinstance(n_jobs, (type(None), int)), print(f"n_jobs was type '{type(n_jobs)}', but must be None or an int")
    assert isinstance(joblib_kwargs, (type(None), dict)), print(f"joblib_kwargs was type '{type(joblib_kwargs)}', but must be None or a dict")

    if tri_func in ['spearman', None]:
        tri_func = spearmanr
    elif tri_func == 'pearson':
        tri_func = pearsonr
    
    rng = np.random.default_rng(seed)
    def shuffle_and_tri_func(a, b, this_seed=None):
        rng = np.random.default_rng(this_seed)
        rng.shuffle(b)
        return tri_func(a, b)
    
    if n_jobs is not None:
        with Parallel(n_jobs=n_jobs, **joblib_kwargs) as parallel:
            null_dist = parallel(delayed(shuffle_and_tri_func)(x_n, x_b, this_seed=rng.integers(0,n_iter))
                                for i in range(n_iter))
    else:
        null_dist = []
        for i in range(n_iter):
            null_dist.append(shuffle_and_tri_func(x_n, x_b, this_seed=rng.integers(0,n_iter)))
    null_dist = np.array(null_dist)
    
    if apply_threshold:
        return null_threshold(tri_func(x_n, x_b), null_dist, 
                                tail=tail, **threshold_kwargs)
    else:
        return null_dist
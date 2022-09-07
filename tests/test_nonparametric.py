# test_nonparametric.py

from functools import partial
import logging
import time
import pytest
from contextlib import nullcontext as does_not_raise

import numpy as np
import numpy.testing as np_test
from scipy.stats import truncnorm

from local_intersubject_pkg.nonparametric import (null_threshold, perm_signflip, 
                                                perm_grouplabel)

logger = logging.getLogger(__name__)

which_tail_param = ['upper', 'lower']

def compare_func(d, n, tail):
    if tail=='upper':
        return d > n
    elif tail == 'lower':
        return d < n
    elif tail == 'both':
        return d > n or d < n 


def get_simple_null_dist(data=None, n_iters=25, size=None, alpha=0.05):
    """
    Get manually created null distribution to test null_threshold() with.
    """
    assert n_iters is None or (type(n_iters)==int and n_iters>=25), f"n_iters must either be an int>=25 or None"
    assert size is None or (type(size)==tuple and len(size)==2), \
        f"If size is not None, size must be be a tuple of len==2, but you provided {size}"
    assert isinstance(alpha, float), f"alpha was type '{type(alpha)}', but must be an float"
    if size is not None:
        n_iters = size[0]
    
    # Get number of true vals that will be P(true null) < pval
    n_sig = np.floor(alpha * n_iters).astype(int)
    
    null = np.array([1,0])
    null = np.repeat(null[None, :], n_iters, axis=0)
    null[:n_sig,1] = np.repeat(2, n_sig)
    return null


@pytest.fixture
def fxt_get_simple_null_dist():
    return get_simple_null_dist


def ref_null(data, stat_mask, n_iters=25, alpha=0.05, null_type='full', tail='upper'):
    """
    Create an idea null distribution that will produce significant values
    that align with data and stat_mask.

    Intended for use when testing null_threshold() parametrically.

    data and stat_mask both must be 1-dimensional array-lik objects.
    """
    
    assert len(data) == len(stat_mask), f"data.shape[0] (n_stats) and len(stat_mask) were not equal: {data.shape} != {len(stat_mask)}"
    assert isinstance(data, (list, np.ndarray)) and isinstance(stat_mask, (list, np.ndarray)),\
        f"Both data and stat_mask must be type list or np.ndarray, but were '{type(data)}' and '{type(stat_mask)}' instead"
    assert n_iters is None or (type(n_iters)==int and n_iters>=25), f"n_iters must either be an int>=25 or None"
    # assert size is None or (type(size)==tuple and len(size)==2), \
    #     f"If size is not None, size must be be a tuple of len==2, but you provided {size}"
    assert isinstance(alpha, float), f"alpha was type '{type(alpha)}', but must be an float"
    assert null_type in ['full', 'max_stat'], f"null_type was '{null_type}', but must be 'full' or 'max_stat'"
    
    logger.debug(f"Setting up ref_null(n_iters={n_iters}, alpha={alpha}, null_type={null_type}, tail={tail})")
    
    # if size is not None:
    #     n_iters = size[0]

    logger.debug(f"data shape={data.shape}\nstat_mask shape={stat_mask.shape}")

    null_ct = np.floor(alpha * n_iters).astype(int)
    logger.debug(f"\nalpha={alpha}\nn_iters={n_iters}\nnull_ct={null_ct}")

    if null_type == 'full':
        if tail == 'upper':
            null_true = data[None,:] + 1
            null_false = data[None,:] - 1
        elif tail == 'lower':
            null_true = data[None,:] - 1
            null_false = data[None,:] + 1
        logger.debug(f"\nnull_true shape={null_true.shape}\nnull_fals shape={null_false.shape}")
        
        null = np.repeat(
            np.where(stat_mask==True, null_false, null_true), n_iters,
            axis=0)
        # null = np.repeat(null_true, n_iters, axis=0)
        logger.debug(f"null(shape:{null.shape})=\n{null}")
        null[:null_ct] = np.repeat(
            np.where(stat_mask==True, null_true, null_true), null_ct,
            axis=0)
        logger.debug(f"null:\n{null}")

    elif null_type == 'max_stat':
        sig_vals = data[stat_mask]
        nonsig_vals = data[~stat_mask]
        if tail == 'upper':
            extreme_null_true = sig_vals.max() + 1
            extreme_null_false = (sig_vals.min() + nonsig_vals.max()) / 2
        elif tail == 'lower':
            extreme_null_true = sig_vals.min() - 1
            extreme_null_false = (sig_vals.max() + nonsig_vals.min()) / 2
        logger.debug(f"\nextreme_null_true={extreme_null_true}\nextreme_null_false={extreme_null_false}")

        null = np.full(shape=(n_iters, len(data)), fill_value=extreme_null_false)
        null[:null_ct] = np.full(shape=(null_ct, len(data)), fill_value=extreme_null_true)
        logger.debug(f"null:\n{null}")

    return null


@pytest.fixture
def fxt_ref_null():
    return ref_null    


def ref_observed(data=None, stat_mask=None, n_iters=25, alpha=0.05):
    """
    Provide a 1-dimensional data and stat_mask array and return
    expected outputs of null_threshold() for different 'return_type' kwarg values.
    """
    
    assert isinstance(data, np.ndarray), f"data was type '{type(data)}', but should be np.ndarray"
    assert isinstance(stat_mask, (list, np.ndarray)), f"stat_mask was type '{type(stat_mask)}', but should be list or np.ndarray"
    logger.debug(f"Setting up ref_observed()")

    # assume upper tail test only atm    
    null_ct = np.floor(alpha * n_iters).astype(int)
    ref = {}
    ref['data'] = data
    # This makes non-sig voxels all True across iterations,
    # but sig voxels only True where rows[:null_ct]
    null_mask = np.full(shape=(n_iters, len(data)), fill_value=~stat_mask)
    logger.debug(f"null_mask shape={null_mask.shape}")
    null_mask[:null_ct] = np.full(shape=(null_ct, len(data)), fill_value=True)
    # null_mask[:null_ct] = np.repeat(
    #     True, null_ct
    # )
    ref['null_mask'] = null_mask
    ref['null_ct'] = np.where(stat_mask==True, null_ct, n_iters)
    ref['pvals'] = ref['null_ct'] / n_iters
    ref['stat_mask'] = stat_mask
    ref['sig_vals'] = np.where(stat_mask==True, data, np.nan)
    return ref


@pytest.fixture
def fxt_ref_observed():
    return ref_observed


def ref_simple_observed(n_iters=25):
    """
    Get manually created observed stats to test null_threshold() with.
    """
    ref = {}
    ref['data'] = np.array([0,1])
    ref['null_mask'] = np.repeat([[True, False]], 25, axis=0)
    ref['null_mask'][0] = [True, True]
    ref['null_ct'] = [25, 1]
    ref['pvals'] = [1, 1/n_iters]
    ref['stat_mask'] = [False, True]
    ref['sig_vals'] = [np.nan, 1]
    return ref


@pytest.fixture
def fxt_ref_simple_observed():
    return ref_simple_observed


def ref_subject_stats(null_lohi=(-0.3, 0.3), sig_lohi=(0.31, 0.9), shape=(1000, 100), 
                    n_sig_embed=0.10, tail='upper', random_seed=None):
    """
    Obtain random subject-level stat values truncated with scipy.stats.truncnorm.rvs,
    and embedded with significantly different values.

    Intended for use when testing perm_signflip()

    Parameters
    ----------
    null_lohi : tuple of floats, default=(-0.3, 0.3)
        The lower and upper range within which most 'null' random values 
        will be created. The number of null values will be shape[0] - n_sig_embed
        (assuming n_sig_embed is transformed into an int)

    sig_lohi : tuple of floats, default=(0.31, 0.9)
        The lower and upper range within which embedded signal stat values
        will be created. The number of null values will be n_sig_embed 
        (or n_sig_embed * n_subjects if it is a float).

        If tail='lower', then sig_lohi = (-sig_lohi[1], -sig_lohi[0])

    shape : tuple of ints, default=(1000, 100)
        The dimensions for the resulting data. Shape=(n_stats, n_subjects)

    n_sig_embed : int, float, default=0.10
        The number of signal stats to embed in the final data. 

        If n_sig_embed is a float, it is multiplied by n_subjects and rounded
        to get an integer of signal stats to embed.

    tail : str, default='upper'
        Choose whether to embed sig_lohi random values toward the upper or lower
        end of the distribution (across all subjects).

        Options:
            - 'upper': apply sig_lohi as-is
            - 'lower': multiply sig_lohi by -1, then switch idx 0 and 1 to
                create embedded signal values toward the lower end
                of the distribution.


    random_seed : None or int, default=None
        Choose the seed used to create random values; allows you generate
        replicable random data.
    """
    assert type(null_lohi)==tuple and type(sig_lohi)==tuple and type(shape)==tuple, \
        print(f"Both null_lohi, sig_lohi, and shape must be tuples, but instead were type '{type(null_lohi)}', '{type(sig_lohi)}', and '{type(shape)}'")
    assert type(shape[0])==int and type(shape[1])==int, f"shape must be a tuple of integers, but you provided {shape}"
    # assert type(n_subjects)==int, f"n_subjects was type '{type(n_subjects)}', but must be an int"
    # assert type(n_stats)==int, f"n_stats was type '{type(n_stats)}', but must be an int"
    assert isinstance(n_sig_embed, (int, float)), \
        f"n_sig_embed was type '{type(n_sig_embed)}', but must be an int or float"
    assert tail in which_tail_param, f"tail was '{tail}', but must be in {which_tail_param}"

    n_stats, n_subjects = shape[0], shape[1]
    if type(n_sig_embed) == float:
        n_sig_embed = np.round(n_sig_embed * n_stats).astype(int)
    assert n_sig_embed < n_stats, \
        f"n_sig_embed ({n_sig_embed}) must be smaller than n_stats ({n_stats})"
    # logger.debug(f"n_sig_embed={n_sig_embed}")
    if tail == 'lower': # reverse numbers if lower tail
        sig_lohi = (-sig_lohi[1], -sig_lohi[0])

    rng = np.random.default_rng(random_seed)
    stats = truncnorm.rvs(null_lohi[0], null_lohi[1], size=(n_stats, n_subjects),
                            random_state=rng)
    # logger.debug(f"stats with null={stats}")
    stats[: n_sig_embed] = truncnorm.rvs(sig_lohi[0], sig_lohi[1], 
                            size=(n_sig_embed, n_subjects), random_state=rng)
    # logger.debug(f"stats with sig embed={stats}")
    return stats


@pytest.fixture
def fxt_ref_subject_stats():
    return ref_subject_stats


# =============
# ====Tests====
# =============


class TestNullThreshold:
    """
    Tests for local_intersubject_pkg.nonparametric.null_threshold()
    """

    return_args = ['null_mask', 'null_ct', 'pvals', 'stat_mask', 'sig_vals']
    n_iters_param = [25, 101, 1001]
    stat_size_param = [10, 100, 1000]
    mask_cutoff_param = [0.23, 0.55, 0.77]

    def assert_null_data(self, ref_data, obs_null):
        assert type(ref_data)==dict and type(obs_null)==dict, \
            f"ref_data and obs_null were types '{type(ref_data)}' and '{type(obs_null)}', but should both be dict"
        """Helper method to test different return types for null_threshold()"""
        logger.debug(f"Running self.assert_null_data() method")

        return_args = self.return_args

        for arg in return_args[1:]:
            logger.debug(f"Check arg={arg}")
            r = ref_data[arg]
            o = obs_null[arg]
            logger.debug(f"ref=\n{r}")
            logger.debug(f"obs=\n{o}")
            np_test.assert_array_equal(r, o)

        if 'null_mask' in list(obs_null.keys()):
            # Test null mask
            logger.debug(f"Checking null_mask arg")
            # for null_idx, r in enumerate(ref_data['null_mask']):
            r = ref_data['null_mask']
            o = obs_null['null_mask']
            logger.debug(f"ref=\n{r}")
            logger.debug(f"obs=\n{o}")
            np_test.assert_array_equal(r, o)

    def test_basic(self, fxt_ref_simple_observed, fxt_get_simple_null_dist):
        """
        Check that outputs are correct for regular null distribution (non-max stat corrected)
        with manually created test data
        """
        logger.debug(f"Running TestNullThreshold().test_basic()")
        # Create reference data
        n_iters = 25
        alpha = 0.05

        ref_data = fxt_ref_simple_observed(n_iters=n_iters)
        ref_null = fxt_get_simple_null_dist(n_iters, alpha=alpha)
        return_args = self.return_args
        obs_null = {arg: null_threshold(ref_data['data'], ref_null, return_type=arg) for arg in return_args}

        self.assert_null_data(ref_data, obs_null)

    @pytest.mark.parametrize('n_iters', n_iters_param)
    @pytest.mark.parametrize('stat_size', stat_size_param)
    @pytest.mark.parametrize('mask_cutoff', mask_cutoff_param)
    @pytest.mark.parametrize('which_tail', which_tail_param)
    def test_full_null(self, n_iters, stat_size, mask_cutoff, which_tail, 
                        fxt_ref_observed, fxt_ref_null):
        """
        Check that function works with a range of null_distribution iterations,
        stat data size, significant values, and test tails.
        """
        
        logger.debug("Running TestNullThreshold().test_full_null()")
        n_iters = n_iters
        alpha = 0.05

        rng = np.random.default_rng(seed=0)
        stats = rng.normal(size=(stat_size))
        stat_mask = compare_func(stats, mask_cutoff, which_tail) # random stat mask

        ref_data = fxt_ref_observed(stats, stat_mask, n_iters=n_iters, alpha=alpha)
        ref_null = fxt_ref_null(stats, stat_mask, n_iters, alpha=alpha, tail=which_tail)

        return_args = self.return_args
        obs_null = {arg:null_threshold(ref_data['data'], ref_null, tail=which_tail, return_type=arg) 
                    for arg in return_args}
        
        self.assert_null_data(ref_data, obs_null)
        
    @pytest.mark.parametrize('n_iters', n_iters_param)
    @pytest.mark.parametrize('stat_size', stat_size_param)
    @pytest.mark.parametrize('mask_cutoff', mask_cutoff_param)
    @pytest.mark.parametrize('which_tail', which_tail_param)        
    def test_max_stat_null(self, n_iters, stat_size, mask_cutoff, which_tail,
                            fxt_ref_observed, fxt_ref_null):
        logger.debug(f"Running TestNullThreshold().test_max_stat_null()")
        n_iters = n_iters
        alpha = 0.05
        rng = np.random.default_rng(seed=100)
        stats = rng.normal(size=(stat_size))
        stat_mask = compare_func(stats, mask_cutoff, which_tail)

        ref_data = fxt_ref_observed(stats, stat_mask, n_iters=n_iters, alpha=alpha)
        ref_null = fxt_ref_null(stats, stat_mask, n_iters, alpha=alpha, 
                                tail=which_tail, null_type='max_stat')

        return_args = self.return_args.copy()
        return_args.remove('null_mask')
        obs_null = {arg:null_threshold(ref_data['data'], ref_null, tail=which_tail, max_stat=True, return_type=arg)
                    for arg in return_args}

        self.assert_null_data(ref_data, obs_null)

    @pytest.mark.parametrize('data, null, expectation',[
        (np.array([0,1,2]), np.array([[0,1,2],[5,5,5]]), does_not_raise()),
        (np.array([0,1,2]), np.array([0,1,2]), does_not_raise()),
        (np.array([0,1,2]), np.array([[0,1,2]]), does_not_raise()),
        (np.array([0,1,2]), np.array([0,1,2,3]), pytest.raises(AssertionError)),
        (np.array([0,1,2]), np.array([[0,1,2,3],[5,5,5,5]]), pytest.raises(AssertionError))
    ])
    def test_shape_mismatch(self, data, null, expectation):
        """
        Check that compatible data shapes do not raise exceptions and mismatched
        shapes raise ValueError
        """
        with expectation:
            assert null_threshold(data, null) is not None


class TestPermSignflip:
    n_iters_param = [
        # 100,
        1000
        ]
    n_subs_and_stats_param = [
        (100, 1000),
        (300, 1000),
        (500, 10_000)
    ]
    sig_prop_param = [0.10, 0.50]
    
    @pytest.mark.parametrize('n_iters', n_iters_param)
    @pytest.mark.parametrize('n_subs,n_stats', n_subs_and_stats_param)
    @pytest.mark.parametrize('sig_prop', sig_prop_param)
    @pytest.mark.parametrize('which_tail', which_tail_param)
    def test_full_null(self, n_iters, n_subs, n_stats, sig_prop, which_tail,
                        fxt_ref_subject_stats):
        logger.debug(f"Running TestPermSignflip().test_full_null()")
        
        # Setup test data
        alpha = 0.05
        stats = fxt_ref_subject_stats(shape=(n_subs, n_stats), n_sig_embed=sig_prop,
                                    tail=which_tail, random_seed=0)

        # Run permutation test and get thresholded data
        n_jobs = -4
        tic = time.time()
        perm_results = perm_signflip(stats, n_iter=n_iters, tail=which_tail,
                                    apply_threshold=True, n_jobs=n_jobs,
                                    threshold_kwargs={'alpha':alpha})
        toc = time.time()

        logger.debug(f"perm_signflip 'full' took {toc-tic:.3f} s with n_jobs={n_jobs}")
        logger.debug(f"perm results shape = {perm_results.shape}")
        perm_n_sig = (~np.isnan(perm_results)).sum()
        logger.debug(f"perm_n_sig={perm_n_sig}, {perm_n_sig/n_stats}")
        
        expected_sig_embed = np.round(sig_prop*n_stats).astype(int)
        expected_null = (alpha * n_stats)

        logger.debug(f"expected sig embed={expected_sig_embed}")
        logger.debug(f"expected null={expected_null}")
        logger.debug(f"perm_n_sig minus expected null={perm_n_sig - expected_null}")
        logger.debug(f"perm_n_sig minus expected sig={perm_n_sig - expected_sig_embed}")
        assert expected_null >= perm_n_sig - expected_sig_embed
        assert expected_sig_embed >= perm_n_sig - expected_null

    @pytest.mark.parametrize('n_iters', n_iters_param)
    @pytest.mark.parametrize('n_subs,n_stats', n_subs_and_stats_param)
    @pytest.mark.parametrize('sig_prop', sig_prop_param)
    @pytest.mark.parametrize('which_tail', which_tail_param)
    def test_max_stat_null(self, n_iters, n_subs, n_stats, sig_prop, which_tail,
                            fxt_ref_subject_stats):
        logger.debug(f"Running TestPermSignflip().test_max_stat_null()")

        alpha = 0.05
        stats = fxt_ref_subject_stats(shape=(n_subs, n_stats), n_sig_embed=sig_prop,
                                    tail=which_tail, random_seed=0)

        n_jobs = -4
        tic = time.time()
        perm_results = perm_signflip(stats, n_iter=n_iters, tail=which_tail,
                                    apply_threshold=True, n_jobs=n_jobs,
                                    threshold_kwargs={'alpha':alpha, 'max_stat':True})
        toc = time.time()

        logger.debug(f"perm_signflip 'max_stat' took {toc-tic:.3f} s with n_jobs={n_jobs}")
        logger.debug(f"perm results shape = {perm_results.shape}")
        perm_n_sig = (~np.isnan(perm_results)).sum()
        logger.debug(f"perm_n_sig={perm_n_sig}, {perm_n_sig/n_stats}")
        
        expected_sig_embed = np.round(sig_prop*n_stats).astype(int)
        expected_null = (alpha * n_stats)

        logger.debug(f"expected sig embed={expected_sig_embed}")
        logger.debug(f"expected null={expected_null}")
        logger.debug(f"perm_n_sig minus expected null={perm_n_sig - expected_null}")
        logger.debug(f"perm_n_sig minus expected sig={perm_n_sig - expected_sig_embed}")
        assert expected_null >= perm_n_sig - expected_sig_embed
        assert expected_sig_embed >= perm_n_sig - expected_null

    def test_signflip_is_subjectwise(self):
        """
        Confirm that signflipping only occurs across subjects (rather than across
        samples or elementwise, for example)
        """
        logger.debug(f"Running TestPermSignflip().test_subject_wise_permutation()")
        
        # Create basic data of ones
        seed = 0
        logger.debug(f"random seed = {seed}")
        data = np.ones(shape=(5,10))

        # Precomputed subject-wise 'flipper' array
        # also generated with rng.choice([-1,1], size=(data.shape[1]))
        # and rng.seed=0
        ref_flipper = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, 1])

        # Check that len of ref_flipper equal to n_subjects from 'data'
        assert data.shape[1] == ref_flipper.shape[0]

        # Check that observed flipped data is the same as 
        # reference flipped data
        ref_flip_data = data * ref_flipper
        logger.debug(f"data:\n{data}")
        logger.debug(f"ref_flipper:\n{ref_flipper}")
        logger.debug(f"ref_flip_data:\n{ref_flip_data}")

        return_func = lambda x: x # x is just returned basically
        obs_flip_data = perm_signflip(data, stat_func=return_func, 
                                    n_iter=1, seed=seed)[0] # un-nest first iteration results
        logger.debug(f"obs_flip_data:\n{obs_flip_data}")

        np_test.assert_array_equal(ref_flip_data, obs_flip_data)

        # Check that masked out +1 and masked out -1 values are equal
        # based on if their sums are the same
        positive_mask = np.where(ref_flipper==1, True, False)
        negative_mask = np.where(ref_flipper==-1, True, False)

        logger.debug(f"+1 subjects mask:\n{positive_mask}")
        logger.debug(f"-1 subjects mask:\n{negative_mask}")

        ref_n_pos = np.sum(ref_flip_data[:,positive_mask])
        ref_n_neg = np.sum(ref_flip_data[:,negative_mask])
        logger.debug(f"ref n pos={ref_n_pos}; ref n neg={ref_n_neg}")

        obs_n_pos = np.sum(obs_flip_data[:,positive_mask])
        obs_n_neg = np.sum(obs_flip_data[:,negative_mask])
        logger.debug(f"obs n pos={obs_n_pos}\nobs n neg={obs_n_neg}")

        # Check that the count of positive and negative ones are equal
        assert ref_n_pos == obs_n_pos
        assert ref_n_neg == obs_n_neg
        

    @pytest.mark.parametrize('n_jobs', [1, 5, -4])
    def test_parallel_equality(self, n_jobs, fxt_ref_subject_stats):
        logger.debug(f"Running TestPermSignflip().test_parallel_equality()")
        stats = fxt_ref_subject_stats(shape=(200, 5000), n_sig_embed=0.10, random_seed=0)

        n_iters = 50
        seed = 0
        stat_func = partial(np.mean, axis=0)
        loop_perm = perm_signflip(stats, stat_func=stat_func, n_iter=n_iters, seed=seed)
        pll_perm = perm_signflip(stats, stat_func=stat_func, n_iter=n_iters, seed=seed, n_jobs=n_jobs)
        np_test.assert_array_almost_equal(loop_perm, pll_perm)

        loop_thresh = null_threshold(stat_func(stats), loop_perm)
        pll_thresh = null_threshold(stat_func(stats), loop_perm)
        logger.debug(f"avg diff between loop and pll thresh results={np.nanmean(loop_thresh-pll_thresh):.4f}")
        np_test.assert_array_almost_equal(loop_thresh, pll_thresh)


class TestPermGrouplabel:
    n_iters_param = [
        # 100,
        1000
        ]
    n_subs_and_stats_param = [
        (100, 1000),
        (300, 1000),
        (500, 10_000)
    ]
    sig_prop_param = [0.10, 0.50]

    @pytest.mark.parametrize('n_iters', n_iters_param)
    @pytest.mark.parametrize('n_subs,n_stats', n_subs_and_stats_param)
    @pytest.mark.parametrize('sig_prop', sig_prop_param)
    @pytest.mark.parametrize('which_tail', which_tail_param)
    def test_full_null(self, n_iters, n_subs, n_stats, sig_prop, which_tail,
                        fxt_ref_subject_stats):

        alpha = 0.05
        n_jobs = -2
        mean_diff_func = lambda x, y: (x-y).mean(axis=0)

        rng = np.random.default_rng(seed=100)
        d1 = fxt_ref_subject_stats(null_lohi=(-0.3,0.3), sig_lohi=(0.3,0.5), 
                                    shape=(n_subs, n_stats), n_sig_embed=sig_prop,
                                    random_seed=rng.integers(0,10))
        d2 = fxt_ref_subject_stats(null_lohi=(-0.3,0.3), sig_lohi=(-0.3,0.0),
                                    shape=(n_subs, n_stats), n_sig_embed=sig_prop,
                                    random_seed=rng.integers(0,10))
        
        tic = time.time()
        perm_results = perm_grouplabel(d1, d2, mean_diff_func,
                                    n_iter=1000, n_jobs=n_jobs, tail=which_tail,
                                    apply_threshold=True,
                                    threshold_kwargs={'alpha':alpha})
        toc = time.time()

        logger.debug(f"perm_grouplabel 'full' took {toc-tic:.3f} s with n_jobs={n_jobs}")
        logger.debug(f"perm results shape = {perm_results.shape}")
        perm_n_sig = (~np.isnan(perm_results)).sum()
        logger.debug(f"perm_n_sig={perm_n_sig}, {perm_n_sig/n_stats}")
        
        expected_sig_embed = np.round(sig_prop*n_stats).astype(int)
        expected_null = (alpha * n_stats)

        logger.debug(f"expected sig embed={expected_sig_embed}")
        logger.debug(f"expected null={expected_null}")
        logger.debug(f"perm_n_sig minus expected null={perm_n_sig - expected_null}")
        logger.debug(f"perm_n_sig minus expected sig={perm_n_sig - expected_sig_embed}")
        assert expected_null >= perm_n_sig - expected_sig_embed
        assert expected_sig_embed >= perm_n_sig - expected_null        

        


if __name__ == "__main__":
    TestNullThreshold()
    TestPermSignflip()
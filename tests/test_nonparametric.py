# test_nonparametric.py

import logging
import pytest
from contextlib import nullcontext as does_not_raise

import numpy as np
import numpy.testing as np_test

from local_intersubject_pkg.nonparametric import null_threshold

logger = logging.getLogger(__name__)

def compare_func(d, n, tail):
    if tail=='upper':
        return d > n
    elif tail == 'lower':
        return d < n
    elif tail == 'both':
        return d > n or d < n 

def get_simple_null_dist(data=None, n_iters=25, size=None, alpha=0.05):
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
        null[:null_ct] = np.repeat(
            np.repeat(extreme_null_true, len(data))[None,:], 
            null_ct, axis=0)
        logger.debug(f"null:\n{null}")

    return null


@pytest.fixture
def fxt_ref_null():
    return ref_null    


def ref_observed(data=None, stat_mask=None, n_iters=25, alpha=0.05):
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


class TestNullThreshold:
    """
    Tests for local_intersubject_pkg.nonparametric.null_threshold()
    """

    return_args = ['null_mask', 'null_ct', 'pvals', 'stat_mask', 'sig_vals']

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

    @pytest.mark.parametrize('n_iters', [25, 101, 1001])
    @pytest.mark.parametrize('stat_size', [10, 100, 1000])
    @pytest.mark.parametrize('mask_cutoff', [0.23, 0.55, 0.77])
    @pytest.mark.parametrize('which_tail', ['upper', 'lower'])
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
        
    @pytest.mark.parametrize('n_iters', [25, 101, 1001])
    @pytest.mark.parametrize('stat_size', [10, 100, 1000])
    @pytest.mark.parametrize('mask_cutoff', [0.23, 0.55, 0.77])
    @pytest.mark.parametrize('which_tail', ['upper', 'lower'])        
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
        (np.array([0,1,2]), np.array([0,1,2,3]), pytest.raises(ValueError)),
        (np.array([0,1,2]), np.array([[0,1,2,3],[5,5,5,5]]), pytest.raises(ValueError))
    ])
    def test_shape_mismatch(self, data, null, expectation):
        """
        Check that compatible data shapes do not raise exceptions and mismatched
        shapes raise ValueError
        """
        with expectation:
            assert null_threshold(data, null) is not None
            



        



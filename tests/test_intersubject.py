# test_intersubject.py

from functools import partial
import numpy as np
import numpy.testing as np_test
from scipy.ndimage import gaussian_filter1d
import logging
import pytest
from local_intersubject_pkg.intersubject import (isc, wmb_isc, finn_isrsa,
                                                dynamic_func, window_generator)
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# Compute ISCs using different input types
# List of subjects with one voxel/ROI
class TestIsc:
    """Test that isc() works with list or np.array input type for 
    one or more brain regions"""

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    def test_single_region(self, fxt_simulated_timeseries):
        logger.info("Running TestIscInputs.test_single_region()")

        n_subjects = self.n_subjects
        n_TRs = self.n_TRs
        random_state = self.random_state

        data = fxt_simulated_timeseries(n_subjects, n_TRs,
                                    n_voxels=None, data_type='list',
                                    random_state=random_state)
        iscs_list = isc(data, pairwise=False, summary_statistic=None)

        # Array of subjects with one voxel/ROI
        data = fxt_simulated_timeseries(n_subjects, n_TRs,
                                    n_voxels=None, data_type='array',
                                    random_state=random_state)
        iscs_array = isc(data, pairwise=False, summary_statistic=None)

        # Check they're the same
        assert np.array_equal(iscs_list, iscs_array)

    def test_multiple_regions(self, fxt_simulated_timeseries):
        logger.info("Running TestIscInputs.test_multiple_regions()")

        n_subjects = self.n_subjects
        n_TRs = self.n_TRs
        n_voxels = self.n_voxels
        random_state = self.random_state
        
        # List of subjects with multiple voxels/ROIs
        data = fxt_simulated_timeseries(n_subjects, n_TRs,
                                    n_voxels=n_voxels, data_type='list',
                                    random_state=random_state)
        iscs_list = isc(data, pairwise=False, summary_statistic=None)

        # Array of subjects with multiple voxels/ROIs
        data = fxt_simulated_timeseries(n_subjects, n_TRs,
                                    n_voxels=n_voxels, data_type='array',
                                    random_state=random_state)
        iscs_array = isc(data, pairwise=False, summary_statistic=None)

        # Check they're the same
        assert np.array_equal(iscs_list, iscs_array)

        logger.info("Finished testing ISC inputs")


    # Check pairwise and leave-one-out, and summary statistics for ISC
    def test_isc_options(self, fxt_simulated_timeseries):

        # Set parameters for toy time series data
        n_subjects = 20
        n_TRs = 60
        n_voxels = 30
        random_state = 42

        logger.info("Testing ISC options")

        data = fxt_simulated_timeseries(n_subjects, n_TRs,
                                    n_voxels=n_voxels, data_type='array',
                                    random_state=random_state)

        iscs_loo = isc(data, pairwise=False, summary_statistic=None)
        assert iscs_loo.shape == (n_voxels, n_subjects)

        # Just two subjects
        iscs_loo = isc(data[..., :2], pairwise=False, summary_statistic=None)
        assert iscs_loo.shape == (n_voxels,)

        iscs_pw = isc(data, pairwise=True, summary_statistic=None)
        assert iscs_pw.shape == (n_voxels, n_subjects*(n_subjects-1)/2)

        # Check summary statistics
        isc_mean = isc(data, pairwise=False, summary_statistic='mean')
        assert isc_mean.shape == (n_voxels,)

        isc_median = isc(data, pairwise=False, summary_statistic='median')
        assert isc_median.shape == (n_voxels,)

        with pytest.raises(ValueError):
            isc(data, pairwise=False, summary_statistic='min')

        logger.info("Finished testing ISC options")


    # Make sure ISC recovers correlations of 1 and less than 1
    def test_isc_output(self, fxt_correlated_timeseries):

        logger.info("Testing ISC outputs")

        data = fxt_correlated_timeseries(20, 60, noise=0,
                                    random_state=42)
        iscs = isc(data, pairwise=False)
        assert np.allclose(iscs[:2, :], 1., rtol=1e-05)
        assert np.all(iscs[-1, :] < 1.)

        iscs = isc(data, pairwise=True)
        assert np.allclose(iscs[:2, :], 1., rtol=1e-05)
        assert np.all(iscs[-1, :] < 1.)

        logger.info("Finished testing ISC outputs")


    # Check for proper handling of NaNs in ISC
    def test_isc_nans(self, fxt_simulated_timeseries):

        # Set parameters for toy time series data
        n_subjects = 20
        n_TRs = 60
        n_voxels = 30
        random_state = 42

        logger.info("Testing ISC options")

        data = fxt_simulated_timeseries(n_subjects, n_TRs,
                                    n_voxels=n_voxels, data_type='array',
                                    random_state=random_state)

        # Inject NaNs into data
        data[0, 0, 0] = np.nan

        # Don't tolerate NaNs, should lose zeroeth voxel
        iscs_loo = isc(data, pairwise=False, tolerate_nans=False)
        assert np.sum(np.isnan(iscs_loo)) == n_subjects

        # Tolerate all NaNs, only subject with NaNs yields NaN
        iscs_loo = isc(data, pairwise=False, tolerate_nans=True)
        assert np.sum(np.isnan(iscs_loo)) == 1

        # Pairwise approach shouldn't care
        iscs_pw_T = isc(data, pairwise=True, tolerate_nans=True)
        iscs_pw_F = isc(data, pairwise=True, tolerate_nans=False)
        assert np.allclose(iscs_pw_T, iscs_pw_F, equal_nan=True)

        assert (np.sum(np.isnan(iscs_pw_T)) ==
                np.sum(np.isnan(iscs_pw_F)) ==
                n_subjects - 1)

        # Set proportion of nans to reject (70% and 90% non-NaN)
        data[0, 0, :] = np.nan
        data[0, 1, :n_subjects - int(n_subjects * .7)] = np.nan
        data[0, 2, :n_subjects - int(n_subjects * .9)] = np.nan

        iscs_loo_T = isc(data, pairwise=False, tolerate_nans=True)
        iscs_loo_F = isc(data, pairwise=False, tolerate_nans=False)
        iscs_loo_95 = isc(data, pairwise=False, tolerate_nans=.95)
        iscs_loo_90 = isc(data, pairwise=False, tolerate_nans=.90)
        iscs_loo_80 = isc(data, pairwise=False, tolerate_nans=.8)
        iscs_loo_70 = isc(data, pairwise=False, tolerate_nans=.7)
        iscs_loo_60 = isc(data, pairwise=False, tolerate_nans=.6)

        assert (np.sum(np.isnan(iscs_loo_F)) ==
                np.sum(np.isnan(iscs_loo_95)) == 60)
        assert (np.sum(np.isnan(iscs_loo_80)) ==
                np.sum(np.isnan(iscs_loo_90)) == 42)
        assert (np.sum(np.isnan(iscs_loo_T)) ==
                np.sum(np.isnan(iscs_loo_60)) ==
                np.sum(np.isnan(iscs_loo_70)) == 28)
        assert np.array_equal(np.sum(np.isnan(iscs_loo_F), axis=0),
                            np.sum(np.isnan(iscs_loo_95), axis=0))
        assert np.array_equal(np.sum(np.isnan(iscs_loo_80), axis=0),
                            np.sum(np.isnan(iscs_loo_90), axis=0))
        assert np.all((np.array_equal(
                            np.sum(np.isnan(iscs_loo_T), axis=0),
                            np.sum(np.isnan(iscs_loo_60), axis=0)),
                    np.array_equal(
                            np.sum(np.isnan(iscs_loo_T), axis=0),
                            np.sum(np.isnan(iscs_loo_70), axis=0)),
                    np.array_equal(
                            np.sum(np.isnan(iscs_loo_60), axis=0),
                            np.sum(np.isnan(iscs_loo_70), axis=0))))

        data = fxt_simulated_timeseries(n_subjects, n_TRs,
                                    n_voxels=n_voxels, data_type='array',
                                    random_state=random_state)

        # Make sure voxel with NaNs across all subjects is always removed
        data[0, 0, :] = np.nan
        iscs_loo_T = isc(data, pairwise=False, tolerate_nans=True)
        iscs_loo_F = isc(data, pairwise=False, tolerate_nans=False)
        assert np.allclose(iscs_loo_T, iscs_loo_F, equal_nan=True)
        assert (np.sum(np.isnan(iscs_loo_T)) ==
                np.sum(np.isnan(iscs_loo_F)) ==
                n_subjects)

        iscs_pw_T = isc(data, pairwise=True, tolerate_nans=True)
        iscs_pw_F = isc(data, pairwise=True, tolerate_nans=False)
        assert np.allclose(iscs_pw_T, iscs_pw_F, equal_nan=True)

        assert (np.sum(np.isnan(iscs_pw_T)) ==
                np.sum(np.isnan(iscs_pw_F)) ==
                n_subjects * (n_subjects - 1) / 2)


class TestWmbIsc:
    def test_basic(self, fxt_ref_high_pos_neg_corr):
        """
        Check that signficant positive difference signals embedded
        in input data are detected when computing within minus between group isc
        """
        seed = 0
        n_samples = 1000
        n_sig_embed = 5
        base, pos, neg = fxt_ref_high_pos_neg_corr(n_samples, seed=seed)

        rng = np.random.default_rng(seed)
        d1 = rng.normal(size=(n_samples, 30, 5))
        d2 = rng.normal(size=(n_samples, 30, 5))

        for i in range(n_sig_embed):
            d1[:,:n_sig_embed,i] = np.repeat(rng.normal(pos, scale=0.1)[None,:], n_sig_embed, axis=0).T
            d2[:,:n_sig_embed,i] = np.repeat(rng.normal(neg, scale=0.1)[None,:], n_sig_embed, axis=0).T
        
        logger.debug(f"d1 shape = {d1.shape}")
        logger.debug(f"d2 shape = {d2.shape}")

        obs_wmb = wmb_isc(d1, d2,
                        subtract_wmb=True,
                        summary_statistic='mean')
        logger.debug(f"obs wmb:\n{obs_wmb}")

        assert obs_wmb[:n_sig_embed].mean() > 1.5

    def test_within_and_between_isc_not_equal(self):
        """
        Check that within and between group iscs are different when
        two different datasets are given as input
        """
        seed = 0
        n_samples = 1000
        n_subs = 5
        rng = np.random.default_rng(seed)
        d1 = rng.normal(size=(n_samples, 30, n_subs))
        d2 = rng.normal(size=(n_samples, 30, n_subs))

        obs_wmb = wmb_isc(d1, d2)
        logger.debug(f"obs wmb shape={obs_wmb.shape}")
        logger.debug(f"obs within group isc:\n{obs_wmb[...,0].round(3)}")
        logger.debug(f"obs between group isc:\n{obs_wmb[...,1].round(3)}")
        np_test.assert_raises(AssertionError, np_test.assert_array_almost_equal, 
                            d1[...,0], d1[...,1])

        for ax in [0, 1, None]:
            obs_w_mean = obs_wmb[...,0].mean(axis=ax)
            obs_b_mean = obs_wmb[...,1].mean(axis=ax)
            logger.debug(f"Comparing mean on axis={ax}")
            logger.debug(f"obs within group mean:\n{obs_w_mean.round(3)}")
            logger.debug(f"obs between group mean:\n{obs_b_mean.round(3)}")
            np_test.assert_raises(AssertionError, np_test.assert_array_almost_equal, 
                            obs_w_mean, obs_b_mean)

        obs_w_halfA = obs_wmb[:, : n_subs, 0]
        obs_w_halfB = obs_wmb[:, n_subs: , 0]
        obs_b_halfA = obs_wmb[:, : n_subs, 1]
        obs_b_halfB = obs_wmb[:, n_subs: , 1]
        logger.debug(f"obs within group halfA:\n{obs_w_halfA.round(3)}")
        logger.debug(f"obs within group halfB:\n{obs_w_halfB.round(3)}")
        logger.debug(f"obs between group halfA:\n{obs_b_halfA.round(3)}")
        logger.debug(f"obs between group halfB:\n{obs_b_halfB.round(3)}")
        np_test.assert_raises(AssertionError, np_test.assert_array_almost_equal, 
                            obs_w_halfA, obs_w_halfB)
        np_test.assert_raises(AssertionError, np_test.assert_array_almost_equal, 
                            obs_b_halfA, obs_b_halfB)

    def test_within_and_between_isc_are_equal(self):
        """
        Check that each half of within and between group isc 
        are equal when the same data is provided twice
        """
        seed = 0
        n_samples = 1000
        n_subs = 5
        rng = np.random.default_rng(seed)
        d1 = rng.normal(size=(n_samples, 30, n_subs))

        obs_wmb = wmb_isc(d1, d1)
        logger.debug(f"obs_wmb shape={obs_wmb.shape}")
        logger.debug(f"obs within group isc:\n{obs_wmb[...,0].round(3)}")
        logger.debug(f"obs between group isc:\n{obs_wmb[...,1].round(3)}")

        obs_w_halfA = obs_wmb[:, : n_subs, 0]
        obs_w_halfB = obs_wmb[:, n_subs: , 0]
        obs_b_halfA = obs_wmb[:, : n_subs, 1]
        obs_b_halfB = obs_wmb[:, n_subs: , 1]
        logger.debug(f"obs within group halfA:\n{obs_w_halfA.round(3)}")
        logger.debug(f"obs within group halfB:\n{obs_w_halfB.round(3)}")
        logger.debug(f"obs between group halfA:\n{obs_b_halfA.round(3)}")
        logger.debug(f"obs between group halfB:\n{obs_b_halfB.round(3)}")
        np_test.assert_array_equal(obs_w_halfA, obs_w_halfB)
        np_test.assert_array_equal(obs_b_halfA, obs_b_halfB)


class TestFinnIsrsa:
    def test_basic(self, fxt_ref_high_pos_neg_corr):
        rng = np.random.default_rng(0)

        size = (10, 1000)
        sigma = 5
        seed = 0
        base, pos_behav, neg_behav = fxt_ref_high_pos_neg_corr(size[1], sigma, seed)
        
        data = rng.normal(size=size)
        logger.debug(f"""
        data shape={data.shape}
        base shape={base.shape}
        pos_behav shape={pos_behav.shape}
        neg_behav shape={neg_behav.shape}
        """)
        data[0] = base

        pos_isrsa = finn_isrsa(data, pos_behav)
        neg_isrsa = finn_isrsa(data, neg_behav)

        logger.debug(f"""
        pos_isrsa,
        idx=0 : {pos_isrsa[0]:.3f}
        idx=2 : {pos_isrsa[2]:.3f}
        """)
        assert pos_isrsa[0] > 0.8 and pos_isrsa[2] < 0.8

        logger.debug(f"""
        neg_isrsa,
        idx=0 : {neg_isrsa[0]:.3f}
        idx=2 : {neg_isrsa[2]:.3f}
        """)
        assert neg_isrsa[0] < -0.8 and neg_isrsa[2] > -0.8


@pytest.fixture
def ref_res1():
    res = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9]
    ]
    return res


@pytest.fixture
def ref_res2():
    res = [
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14],
        [13, 14, 15],
        [14, 15, 16],
        [15, 16, 17],
        [16, 17, 18],
        [17, 18, 19]
    ]
    return res


class TestDynamicFunc:
    ref_ndims = [
            (100, 50, 30),
            (100, 50, 30, 20),
            (100, 50, 30, 20, 10)
        ]

    @pytest.mark.parametrize('ndims', ref_ndims)
    def test_one_group_multi_dim(self, ndims):

        func = partial(np.mean, axis=0)

        window_size = 5
        rng = np.random.default_rng(seed=0)
        data = rng.normal(size=ndims)
        start_idxs = [i for i in range(ndims[0] - 4)]

        ref_res = []
        for idx in start_idxs:
            res = func(data[idx: idx+window_size])
            ref_res.append(res)
        ref_res = np.array(ref_res)

        obs_res = dynamic_func(func, data, window_size=window_size,
                                    gaussian_filter_mode=None)
        
        logger.debug(f"ref res shape={ref_res.shape}")
        logger.debug(f"obs res shape={obs_res.shape}")
        assert ref_res.shape == obs_res.shape

        logger.debug(f"ref res:\n{ref_res[0,0,0].round(3)}")
        logger.debug(f"obs res:\n{obs_res[0,0,0].round(3)}")
        np_test.assert_array_equal(ref_res, obs_res)

    # @pytest.mark.parametrize('ndims',ref_ndims) 
    # @pytest.mark.parametrize('func_axis', [1, 2])
    @pytest.mark.parametrize('add_filter', [None, 'reflect'])
    def test_one_group_simple_func(self, ref_res1, add_filter):
        """
        Check that output for one dataset returns as expected with simple summing
        function
        """

        return_func = lambda x: np.sum(x)

        data = [i for i in range(10)]
        window_size = 3
        step = 1
        filter_method = add_filter
        sigma = 3

        if add_filter is not None:
            ref_result = [np.sum(gaussian_filter1d(i, sigma, axis=0, mode=filter_method)) 
                                for i in ref_res1]
        else:
            ref_result = [np.sum(i) for i in ref_res1]
        logger.debug(f'data={data}')
        
        obs_result = dynamic_func(return_func, data, window_size=window_size, 
                                    step=step, gaussian_filter_mode=filter_method, sigma=sigma)
        
        logger.debug(f"ref_result=\n{ref_result}")
        logger.debug(f"obs_result shape={obs_result.shape}")
        logger.debug(f"obs_result=\n{obs_result}")

        np_test.assert_array_equal(ref_result, obs_result)

    @pytest.mark.parametrize('add_filter',[None, 'reflect'])
    def test_two_group_simple_func(self, ref_res1, ref_res2, add_filter):
        """
        Check that output for two datasets returns as expected with simple
        summing function
        """
        return_func = lambda x,y: np.sum(x+y)
        d1 = [i for i in range(10)]
        d2 = [i for i in range(10,20)]
        window_size = 3
        step = 1
        ref_res1 = ref_res1
        ref_res2 = ref_res2

        logger.debug(f'd1={d1}')
        logger.debug(f"d2={d2}")
        
        filter_method = add_filter
        sigma = 3

        if add_filter is not None:
            filter = partial(gaussian_filter1d, sigma=sigma, axis=0, mode=filter_method)
            ref_res = []
            for i, j in zip(ref_res1, ref_res2):
                i = filter(i)
                j = filter(j)
                ref_res.append(np.sum(i+j))
        else:
            ref_res = [np.sum(i+j) for i,j in zip(ref_res1, ref_res2)]
        
        obs_res = dynamic_func(return_func, d1, d2, window_size=window_size, 
                                    step=step, gaussian_filter_mode=filter_method, sigma=sigma)
        
        logger.debug(f"ref_res1=\n{ref_res1}")
        logger.debug(f"ref_res2=\n{ref_res2}")
        logger.debug(f"ref_res=\n{ref_res}")
        logger.debug(f"obs_result shape={obs_res.shape}")
        logger.debug(f"obs_result=\n{obs_res}")

        np_test.assert_array_equal(ref_res, obs_res)


class TestWindowGenerator:
    def test_one_data(self):
        wind_size = 3
        step = 1
        data = [i for i in range(10)]
        logger.debug(f"data={data}")
        ref_windows = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]
        ]

        obs_windows = [w for w in window_generator(data, window_size=wind_size, step=step)]
        logger.debug(f"obs_windows:\n{obs_windows}")
        for i in range(len(ref_windows)):        
            r = ref_windows[i]
            o = obs_windows[i]
            logger.debug(f"r[{i}] {r}")
            logger.debug(f"o[{i}] {o}")
            assert r == o
            logger.debug(f"r and o were equivlent!")

    def test_two_data(self):
        wind_size = 3
        step = 1
        d1 = [i for i in range(10)]
        d2 = [i for i in range(10,20)]
        logger.debug(f"d1={d1}")
        logger.debug(f"d1={d2}")
        ref_wind1 = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]
        ]
        ref_wind2 = [
            [10, 11, 12],
            [11, 12, 13],
            [12, 13, 14],
            [13, 14, 15],
            [14, 15, 16],
            [15, 16, 17],
            [16, 17, 18],
            [17, 18, 19]
        ]

        obs_windows = [w for w in window_generator(d1, d2, window_size=wind_size, step=step)]
        logger.debug(f"obs_windows:\n{obs_windows}")

        logger.debug("Compare d1,d2 and obs_windows line by line:")
        for i in range(len(ref_wind1)):        
            r1 = ref_wind1[i]
            r2 = ref_wind2[i]
            
            logger.debug(f"obs_windows[{i}] {obs_windows[i]}")
            o1 = obs_windows[i][0]
            o2 = obs_windows[i][1]

            logger.debug(f"r1[{i}] {r1}")
            logger.debug(f"o1[{i}] {o1}")
            assert r1 == o1
            logger.debug(f"r1 and o1 were equivlent!")

            logger.debug(f"r2[{i}] {r2}")
            logger.debug(f"o2[{i}] {o2}")
            assert r2 == o2
            logger.debug(f"r2 and o2 were equivlent!")

if __name__ == '__main__':
    TestIsc()
    logger.info("Finished all ISC tests")
# test_intersubject.py

import numpy as np
import numpy.testing as np_test
import logging
import pytest
from local_intersubject_pkg.intersubject import (isc, wmb_isc, finn_isrsa,
                                                dynamic_func)
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


if __name__ == '__main__':
    TestIsc()
    logger.info("Finished all ISC tests")
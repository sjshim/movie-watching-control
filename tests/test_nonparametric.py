# test_nonparametric.py

import logging
import pytest
from contextlib import nullcontext as does_not_raise

import numpy as np
import numpy.testing as np_test

from local_intersubject_pkg.nonparametric import null_threshold

logger = logging.getLogger(__name__)

class TestNullThreshold:
    def test_basic(self):
        """
        Check that outputs are correct for regular null distribution (non-max stat corrected)
        """
        # Create reference data
        n_iters = 25
        data = np.array([0,1])
        null = np.array([1,0])
        null = np.repeat(null[None, :], n_iters, axis=0)
        null[0,1] = 2

        ref = {}
        ref['null_mask'] = [
            (0, [True, True]),
            (slice(1, None), np.repeat([[True, False]], 24, axis=0))
        ]
        ref['null_ct'] = [25, 1]
        ref['pvals'] = [1, 1/n_iters]
        ref['stat_mask'] = [False, True]
        ref['sig_vals'] = [np.nan, 1]

        return_args = ['null_mask', 'null_ct', 'pvals', 'stat_mask', 'sig_vals']
        obs = {arg: null_threshold(data, null, return_type=arg) for arg in return_args}
        
        # Test null mask
        logger.debug(f"Checking null_mask arg")
        for null_idx, r in ref['null_mask']:
            o = obs['null_mask'][null_idx]
            logger.debug(f"null_idx={null_idx}; answer={r}")
            logger.debug(f"obs null mask[null_idx]={o}")
            np_test.assert_array_equal(r, o)

        for arg in return_args[1:]:
            logger.debug(f"Check arg={arg}")
            r = ref[arg]
            o = obs[arg]
            logger.debug(f"ref={r}")
            logger.debug(f"obs={o}")
            np_test.assert_array_equal(r, o)

    def test_max_stat(self):
        pass


    @pytest.mark.parametrize('data, null, expectation',[
        (np.array([0,1,2]), np.array([[0,1,2],[5,5,5]]), does_not_raise()),
        (np.array([0,1,2]), np.array([0,1,2]), does_not_raise()),
        (np.array([0,1,2]), np.array([[0,1,2]]), does_not_raise()),
        (np.array([0,1,2]), np.array([0,1,2,3]), pytest.raises(ValueError)),
        (np.array([0,1,2]), np.array([[0,1,2,3],[5,5,5,5]]), pytest.raises(ValueError))
    ])
    def test_shape_mismatch(self, data, null, expectation):
        with expectation:
            assert null_threshold(data, null) is not None
            



        



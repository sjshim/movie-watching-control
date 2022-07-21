# conftest.py
# Purpose: common helper functions and data shared across tests functions in 
# this project

import pytest
import numpy as np
from scipy.spatial.distance import squareform

# Create simple simulated data with high intersubject correlation
def simulated_timeseries(n_subjects, n_TRs, n_voxels=30,
                         noise=1, data_type='array',
                         random_state=None):
    prng = np.random.default_rng(random_state)
    if n_voxels:
        signal = prng.standard_normal((n_TRs, n_voxels))
        prng = np.random.RandomState(prng.integers(0, 2**32 - 1))
        data = [signal + prng.standard_normal((n_TRs, n_voxels)) * noise
                for subject in np.arange(n_subjects)]
    elif not n_voxels:
        signal = prng.standard_normal(n_TRs)
        prng = np.random.RandomState(prng.integers(0, 2**32 - 1))
        data = [signal + prng.standard_normal(n_TRs) * noise
                for subject in np.arange(n_subjects)]
    if data_type == 'array':
        if n_voxels:
            data = np.dstack(data)
        elif not n_voxels:
            data = np.column_stack(data)
    return data


# Create 3 voxel simulated data with correlated time series
def correlated_timeseries(n_subjects, n_TRs, noise=0,
                          random_state=None):
    prng = np.random.default_rng(random_state)
    signal = prng.standard_normal(n_TRs)
    correlated = True
    while correlated:
        uncorrelated = prng.standard_normal((n_TRs,
                                       n_subjects))[:, np.newaxis, :]
        unc_max = np.amax(squareform(np.corrcoef(
            uncorrelated[:, 0, :].T), checks=False))
        unc_mean = np.mean(squareform(np.corrcoef(
            uncorrelated[:, 0, :].T), checks=False))
        if unc_max < .25 and np.abs(unc_mean) < .001:
            correlated = False
    data = np.repeat(np.column_stack((signal, signal))[..., np.newaxis],
                     n_subjects, axis=2)
    data = np.concatenate((data, uncorrelated), axis=1)
    data = data + prng.standard_normal((n_TRs, 3, n_subjects)) * noise
    return data


def pytest_configure():
    pytest.simulated_timeseries = simulated_timeseries
    pytest.correlated_timeseries = correlated_timeseries
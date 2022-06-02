# bnk_isc_utils.py
# Purpose: Reimplemented Brainiak functions in order to avoid OS installation
# limitations.
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ====================================
# Functions taken from brainiak/isc.py
# ====================================


def _threshold_nans(data, tolerate_nans):

    """Thresholds data based on proportion of subjects with NaNs
    Takes in data and a threshold value (float between 0.0 and 1.0) determining
    the permissible proportion of subjects with non-NaN values. For example, if
    threshold=.8, any voxel where >= 80% of subjects have non-NaN values will
    be left unchanged, while any voxel with < 80% non-NaN values will be
    assigned all NaN values and included in the nan_mask output. Note that the
    output data has not been masked and will be same shape as the input data,
    but may have a different number of NaNs based on the threshold.
    Parameters
    ----------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data
    tolerate_nans : bool or float (0.0 <= threshold <= 1.0)
        Proportion of subjects with non-NaN values required to keep voxel
    Returns
    -------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data with adjusted NaNs
    nan_mask : ndarray (n_voxels,)
        Boolean mask array of voxels with too many NaNs based on threshold
    """

    nans = np.all(np.any(np.isnan(data), axis=0), axis=1)

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans is True:
        logger.info("ISC computation will tolerate all NaNs when averaging")

    elif type(tolerate_nans) is float:
        if not 0.0 <= tolerate_nans <= 1.0:
            raise ValueError("If threshold to tolerate NaNs is a float, "
                             "it must be between 0.0 and 1.0; got {0}".format(
                                tolerate_nans))
        nans += ~(np.sum(~np.any(np.isnan(data), axis=0), axis=1) >=
                  data.shape[-1] * tolerate_nans)
        logger.info("ISC computation will tolerate voxels with at least "
                    "{0} non-NaN values: {1} voxels do not meet "
                    "threshold".format(tolerate_nans,
                                       np.sum(nans)))

    else:
        logger.info("ISC computation will not tolerate NaNs when averaging")

    mask = ~nans
    data = data[:, mask, :]

    return data, mask


def compute_summary_statistic(iscs, summary_statistic='mean', axis=None):

    """Computes summary statistics for ISCs
    Computes either the 'mean' or 'median' across a set of ISCs. In the
    case of the mean, ISC values are first Fisher Z transformed (arctanh),
    averaged, then inverse Fisher Z transformed (tanh).
    The implementation is based on the work in [SilverDunlap1987]_.
    .. [SilverDunlap1987] "Averaging correlation coefficients: should
       Fisher's z transformation be used?", N. C. Silver, W. P. Dunlap, 1987,
       Journal of Applied Psychology, 72, 146-148.
       https://doi.org/10.1037/0021-9010.72.1.146
    Parameters
    ----------
    iscs : list or ndarray
        ISC values
    summary_statistic : str, default: 'mean'
        Summary statistic, 'mean' or 'median'
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
    Returns
    -------
    statistic : float or ndarray
        Summary statistic of ISC values
    """

    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Compute summary statistic
    if summary_statistic == 'mean':
        statistic = np.tanh(np.nanmean(np.arctanh(iscs), axis=axis))
    elif summary_statistic == 'median':
        statistic = np.nanmedian(iscs, axis=axis)

    return statistic


# ============================================
# Functions taken from brainiak/utils/utils.py
# ============================================


def _check_timeseries_input(data):

    """Checks response time series input data (e.g., for ISC analysis)
    Input data should be a n_TRs by n_voxels by n_subjects ndarray
    (e.g., brainiak.image.MaskedMultiSubjectData) or a list where each
    item is a n_TRs by n_voxels ndarray for a given subject. Multiple
    input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. This
    function is generally intended to be used internally by other
    functions module (e.g., isc, isfc in brainiak.isc).
    Parameters
    ----------
    data : ndarray or list
        Time series data
    Returns
    -------
    data : ndarray
        Input time series data with standardized structure
    n_TRs : int
        Number of time points (TRs)
    n_voxels : int
        Number of voxels (or ROIs)
    n_subjects : int
        Number of subjects
    """

    # Convert list input to 3d and check shapes
    if type(data) == list:
        data_shape = data[0].shape
        for i, d in enumerate(data):
            if d.shape != data_shape:
                raise ValueError("All ndarrays in input list "
                                 "must be the same shape!")
            if d.ndim == 1:
                data[i] = d[:, np.newaxis]
        data = np.dstack(data)

    # Convert input ndarray to 3d and check shape
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data[:, np.newaxis, :]
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             "or 3 dimensions (got {0})!".format(data.ndim))

    # Infer subjects, TRs, voxels and log for user to check
    n_TRs, n_voxels, n_subjects = data.shape
    logger.info("Assuming {0} subjects with {1} time points "
                "and {2} voxel(s) or ROI(s) for ISC analysis.".format(
                    n_subjects, n_TRs, n_voxels))

    return data, n_TRs, n_voxels, n_subjects


def array_correlation(x, y, axis=0):

    """Column- or row-wise Pearson correlation between two arrays
    Computes sample Pearson correlation between two 1D or 2D arrays (e.g.,
    two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
    between each corresponding column (axis=0) or row (axis=1) where axis
    indexes observations. If axis=0 (default), each column is considered to
    be a variable and each row is an observation; if axis=1, each row is a
    variable and each column is an observation (equivalent to transposing
    the input arrays). Input arrays must be the same shape with corresponding
    variables and observations. This is intended to be an efficient method
    for computing correlations between two corresponding arrays with many
    variables (e.g., many voxels).
    Parameters
    ----------
    x : 1D or 2D ndarray
        Array of observations for one or more variables
    y : 1D or 2D ndarray
        Array of observations for one or more variables (same shape as x)
    axis : int (0 or 1), default: 0
        Correlation between columns (axis=0) or rows (axis=1)
    Returns
    -------
    r : float or 1D ndarray
        Pearson correlation values for input variables
    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Check that inputs are same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays must be the same shape")

    # Transpose if axis=1 requested (to avoid broadcasting
    # issues introduced by switching axis in mean and sum)
    if axis == 1:
        x, y = x.T, y.T

    # Center (de-mean) input variables
    x_demean = x - np.mean(x, axis=0)
    y_demean = y - np.mean(y, axis=0)

    # Compute summed product of centered variables
    numerator = np.sum(x_demean * y_demean, axis=0)

    # Compute sum squared error
    denominator = np.sqrt(np.sum(x_demean ** 2, axis=0) *
                          np.sum(y_demean ** 2, axis=0))

    return numerator / denominator
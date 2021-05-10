# basic_stats.py: Basic stats functions (not available in Numpy alone) that are repeatedly used in the other fancier
# modules in this package.

import numpy as np

# =========================
# Basic statistics functions
# =========================
# Function for fisher transformation; used for one correlation coefficent averaging method
def fisher_transform(x, output_type='z', method='short'):
    """
    Calculates regular Fisher z-transformation of Pearson correlation 
    coefficients (aka r values) into z-scores, or inverse fisher 
    z-transformation of z-scores into Pearson correlation coefficients.

    Parameters
    ----------
    x : array-like
        The data array to be transformed.
    
    output_type : str
        The output statistic that is desired. 'z' implies regular 
        Fisher z-transformation, while 'r' implies the inverse.

        (Default is 'z')
    
    method : str
        (Primarily exists for function test purposes)
        Chooses the method of calculation. 'short' uses numpy.arctanh and
        numpy.tanh to produce regular and inverse transformations,
        respectively. 'Long' instead calculates it step by step, but still
        uses numpy.log and numpy.exp for regular and inverse transforms.
        
        (Default is 'short')

    Returns
    -------
    output : array-like
        Will have the same shape/dimensions as x


    Warning:
    Using Fisher-transformation in within-between ISC (for calculating)
    average z then inverse transforming back to r) currently produces
    sporadic missing data, so the averaging method should use the
    median instead until this the effect of this function on results
    is better tested and validated.

    """
    # Currently, both output types assume the correct input (i.e., r or z) is provided
    # Transform Pearson r correlation coefficients into z by default or r when defined
    def arctanh_tanh(x, output): # fisher z transformation calculated with arctanh and tanh
        if output == 'z':
            z = np.arctanh(x)
            return z
        elif output == 'r':
            r = np.tanh(x)
            return r

    def long_fisher(x, output): # fisher z transformation calculated the long way
        if output == 'z':
            z = np.log((1+x)/(1-x)) / 2
            return z
        # Transform z scores back into r-values (presumably after z is averaged)
        elif output == 'r':
            r = (np.exp(2*x)-1) / (np.exp(2*x)+1)
            return r

    if method == 'short':
        transformed_output = arctanh_tanh(x, output_type)
    elif method == 'long':
        transformed_output = long_fisher(x, output_type)

    return transformed_output

# Retrieves upper triangle of a matrix 
# TODO: I should make this upper or lower option
# NOTE: This function is not currently incorporated in any other function--should it be?
def unique_tri(array):
    """
    Retrieves the upper-triangle of a square m x m matrix; used in the
    context of getting unique x and y pairs. This currently excludes 
    diagonal values since numpy.triu() already includes it.

    Parameters
    ----------
    array : array-like
        The matrix that the triangle is obtained from

    Returns
    -------
    output : the triangle (excluding the diagonal values)

    """

    m = array.shape[0]
    r, c = np.triu_indices(m,1)
    return array[r,c]

# Computes Pearson correlation coefficient between two matrices
def compute_r(a, b, compare_index='pairwise', axis=0):
    """
    Compute Pearsons' R for every x,y combination
    Note: 
    - allows for (i,j) and (j,i)
    - symmetry depends on whether x == y
    - final matrix shape can be non-square
    """

    # rows are vars: change cols to rows if other axis is desired
    if axis==1:
        a, b = a.T, b.T

    # Reshape 1d data into 2d array with 1 row
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]
    
    # Correlate every combination of (x,y) pairs of rows
    if compare_index == 'pairwise':
        r_array = np.full((b.shape[0], a.shape[0]), np.nan)
        for i, b_row in enumerate(b):
            for j, a_row in enumerate(a):
                r_array[i, j] = np.corrcoef(a_row, b_row)[0, 1]

    elif compare_index == 'same':
        shortest_row = sorted([a.shape[0], b.shape[0]])[0] # first item is shortest
        r_array = np.full(shortest_row, np.nan)
        for i, row in enumerate(zip(a, b)):
            r_array[i] = np.corrcoef(row[0], row[1])[0, 1]

    return r_array

# Function to easily choose the averaging method for a matrix of correlation coefficients
def r_average(x, method='median', axis=None, fisher_method='short'):
    """
    Method can be
    - median (default)
    - fisher (fisher z -> mean -> fisher z inverse)
    - mean (not recommended, but included for comparison purposes)
    """
    if method=='median':
        r_average = np.median(x, axis=axis)
    
    elif method=='fisher':
        if fisher_method=='short': # uses arctanh / tanh (inverse transform)
            r_average = np.mean(fisher_transform(x), axis=axis) # get mean z values
            r_average = fisher_transform(r_average, output_type='r') # turn mean z-values back into r

        elif fisher_method=='long': # calculates transformation the long way
            r_average = np.mean(fisher_transform(x, method='long'), axis=axis)
            r_average = fisher_transform(r_average, output_type='r', method='long')
    
    elif method=='mean':
        r_average = np.mean(x, axis=axis)

    return r_average
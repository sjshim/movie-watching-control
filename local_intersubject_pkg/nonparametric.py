# nonparametric.py: This module contains functions for nonparametric assessment
# and thresholding of fMRI data produced during intersubject analysis.

from multiprocessing import Pool

import numpy as np

from .basic_stats import r_average
from local_intersubject_pkg.intersubject import Intersubject

def tail_null(fake_average, true_average, tail='upper'):
    """
    Assess observed data against a particular tail of a null distribution.

    Used for checking permutation's test at each iteration and to build
    a sum of true null test for each voxel.

    Parameters
    -------
    fake_average:
        1-d array of null distribution averages from iteration i
    
    true_average:
        1-d array of observed averages from data

    tail: 
        - upper
        - lower
        - two
        Note: 'two' reverts to 'upper' because the thresholding function
        properly handle's  

    Returns
    -------
    Optional: array or tuple of arrays: 
        Returns one or two 1-d boolean arrays for a one-tailed or two-tailed test, respectively
        
        True/1 value : null average is more extreme than observed average
        False/0 value : null average is less extreme than observed average
        ("extreme" meaning closer to a distribution's tail ends)
    """
    assert fake_average.shape == true_average.shape, f"both fake {fake_average.shape} and true {true_average.shape} shapes are not the same"

    # Note: Could be simpler, but I really spell it out this way for clarity's sake

    # Force "two" into "upper" tail test
    if tail == 'two':
        tail = 'upper'

    # Get upper or lower tail tests against a null distribution average
    try:
        if tail == 'lower':
            null_left = fake_average < true_average
            return null_left
        elif tail == 'upper':
            null_right = fake_average > true_average
            return null_right
        else:
            raise ValueError(f"'tail' must be either 'lower', or 'upper'; you provided {tail}")
    except:
        print(f"Failed to create null count using {tail}")


def threshold_mask(null_count, n_iter, sig_level, tail='two', output_type='masked_data', data=None):
    """
    Threshold data using results from a nonparametric 
    significance test.

    For each voxel, a p-value is computed by dividing the null count by
    the number of iteratons used to generate it. The threshold mask
    is created to retain voxels whose p-values are smaller than the
    significance level.

    Parameters
    ----------

    null_count : array
        The count of true null hypothesis tests generated from a repetitive
        nonparametric test. 
        
        Note: this function does not check whether this belongs
        to an upper or lower tail test. It will handle the one or two 
        tailed thresholding operations regardless.

    n_iter : int
        The number of iterations used to produce the null count array.

    sig_level : str
        The significance level (or alpha) is used to generate a boolean 
        mask whose values evaluate according to whether the p-value 
        (ie., null count divided by number of test iterations) is lower 
        than the significance level.

        Note: sig_level is assumed to hav been adjusted for a one or two 
        tailed test.

    tail : str
        Options are 'two' or 'one'. This function does not know whether
        one tailed null count array's reflect an upper or lower tail test.

    output_type : str
        Options are 'p_values', 'mask', or 'masked_data'. Masked data is
        default.

    data : array, optional
        The data to be thresholded. Only required if `output_type` is 
        masked_data.



    """
    # Calculate p-values
    p_values = null_count/n_iter

    # Check what threshold-related info to return
    if output_type == 'p_val':
        return p_values

    if tail == 'one':
        mask = p_values < sig_level

    elif tail =='two':
        right_tail = p_values < sig_level
        left_tail = (1 - p_values) < sig_level
        mask = np.logical_or(right_tail, left_tail)

    if output_type == 'mask':
        return mask

    elif output_type == 'masked_data':
        assert data is not None, "'data' must be provided to produced masked_data"
        assert data.shape[0] == mask.shape[0], f"Row size of data {data.shape} and mask {mask.shape} do not match"

        masked_data = np.full(data.shape, np.nan)
        masked_data[mask] = data[mask]
        return masked_data

# 1-sample permutation test with r-value sign flipping
def perm_signflip(x, n_iter, tail='two', average_method='median', shuffle_method='subject', 
                    compute_method='null_count', output_type='masked_data', 
                    sig_level=None, seed=None, x_avg=None):
    """
    Compute permutation test by flipping array values positive or negative 
    for several iterations. Performed randomly either across all 
    element values or by subjects. 

    Parameters
    ----------
    x : array
        The un-averaged dataset to be permuted.

    n_iter : int
        Number of iterations to perform this test

    tail : str
        The null distribution tail to test the data against. 
        Tail can either be 'two', 'upper', or 'lower'.

    average_method : str
        The averaging method to use

    shuffle_method : str
        Can be 'element' or 'subject'

    compute_method : str
        Can be 'null_count' or 'entire_dist'

    output_type : str
        Options: 'p_val', 'mask', 'masked_data', 
        'null_count' or 'entire_dist'

    sig_level : str
        The value to compare permutation test with real average data against.

    seed = int, optional
        Choose the random process seed

    x_avg : array, optional
    Optionally provide the averaged dataset to prevent this function
    from calulating the average for you.

    Returns
    -------
    array
        An array of p-values, a mask, the thresholded data itself, 
    the null count, or the entire null distribution.


    """
    
    
    # NOTE: This is element- (EWP) and subject-wise (SWP) permutation
    # - Russ' fmri handbook says permutation>boostrap
    # - Gang chen et al (2016) says subject wise bootstrap (SWB)>EWP bc of fpr
    # I should change this test or replace with SW booststrap later
    
    #---------
    # Check input argument validity
    assert tail in ['upper', 'lower', 'two'], "Invalid tail choice"
    assert average_method in ['median', 'fisher', 'mean'], "Invalid average method"
    assert compute_method in ['null_count', 'entire_dist'], "Invalid compute method"
    assert shuffle_method in ['element', 'subject'], "Invalid shuffle method"
    assert output_type in ['p_val', 'mask', 'masked_data', 'null_count', 'entire_dist'], "Invalid output type"

    x_matrix = x # saved separately bc problems have occured occasionally 
    x_dims = x_matrix.shape
    assert x_matrix.ndim == 2, f"x must be 2d matrix; {x_matrix.ndim} dim array provided instead"
    
    # Check for x average input
    if x_avg != None:
        true_average = x_avg
    else:    
        true_average = r_average(x_matrix, method=average_method, axis=1)
    
    # Check for seed input
    if seed == None:
        seed = 420
        
    # Check computation method to use
    if compute_method == 'null_count':
        perm_output = np.zeros((x_dims[0])) # make 1d array the size of number of voxels
    elif compute_method == 'entire_dist':
        perm_output = np.zeros((x_dims[0], n_iter)) # make matrix with dimensions of (no. voxels x n_iter)
        output_type = compute_method
    # assert perm_output.dtype is np.float64, f'perm_output dtype is {perm_output.dtype}, not bool'
    # print(f"perm output shape {perm_output.shape}")
        
    #---------
    # Create sign-permuted null distribution
    for i in range(n_iter):
        # print('Perm iteration {}'.format(i))

        # Create array of random 1 and -1 
        np.random.seed(seed + i*100) # explicit random seeds for reproducibility
        
        if shuffle_method == 'element': #randomly flip signs for all ISC values
            sign_flip = np.random.random(x_dims) > 0.5
        elif shuffle_method == 'subject': #for random subset of subjects, flip all ISC signs 
            sign_flip = np.random.random(x_dims[1]) > 0.5
            
        sign_flip = np.where(sign_flip==True, 1, -1) #If true, assign 1; else assign -1
        # assert
            
        # Create null distribution by multiplying data by 1 or -1; then average across voxels
        x_permuted = x_matrix * sign_flip
        fake_average = r_average(x_permuted, method=average_method, axis=1)

        # Upper one-tailed test for each average voxel
        # Null hypothesis: True average is not greater than Null average
        if compute_method=='null_count':
            # return null count depending on tail test arg
            this_count = tail_null(fake_average, true_average, tail)
            # print(this_count.shape, this_count.dtype, this_count)
            # assert this_count.dtype is bool
            perm_output = perm_output + this_count # add rejection counts from current iteration
        
        # Warning: the entire null distribution will probably be a big array!        
        elif compute_method=='entire_dist':
            perm_output[:, i] = fake_average # add null ISC averages from current iteration

    # Return desired output 
    if output_type in ['null_count', 'entire_dist']:
        return perm_output
    else:
        if tail in ['upper', 'lower']:
            tail = 'one'

        threshold_result = threshold_mask(perm_output, n_iter, sig_level, 
                                tail, output_type, data=true_average)
        return threshold_result


# TODO: create this function and make it parallelizable! 
# # 2-sample group label permutation test
def perm_grouplabel(x_avg, data_path, sub_id_dict, datasize, n_iter, 
        tail='two', average_method='median', output_type='masked_data', 
        sig_level=None, seed=None):
    """
    Performs a two-sample group label for intersubject correlation.

    This function computes the permutation test by randomly reassigning 
    subjects between two groups, then recalculating the statistic 
    for many iterations. The resulting null distribution is compared 
    against the real averaged data .

    NOTE: This function currently assumes that within-between ISC is being
    tested, although it could technically be modified to  work for 
    the usual between-group comparison.

    Parameters
    ----------
    x_avg : str
        The average dataset used to test against the null distribution.

    data_path : str
        Filepath to subjects' fMRI data; should have string formatting 
        to allow for looping for subject id's

    sub_id_dict : dict
        Dict of left and right subject ids, which should take the 
        following shape:

        {'left': [list of subject ids],
        'right': [list of subject ids]}

    datasize : list-like
        Global 4d dimensions of the data for intersubject analysis. Numbers
        represent the 3d volume (x, y, z) + the number of repetition times/
        TRs.

    n_iter : int
        Number of iterations that this permutation test should perform

    tail : str
        Choose which distribution tail or tails to test for significance.
        Options include:

        - 'both' : Perform two-tailed test
        - 'lower' or 'upper' : Perform left or right tail test, respectively

    average_method : str
        The averaging method that should be used to calculate each voxel's
        average within-between ISC/ISFC difference for the null distribution

    output_type : str
        The output type that should be returned by the thresholding 
        function. Options are 'masked_data', 'mask', and 'p_val'

    seed : int
        The starting seed to use for randomizing subjects' group labels. 
        If None is chosen, then defaults to 1.

    Returns
    -------
    perm_output : 1d array, with size equal to n subjects included.
        An array reflecting the number of times where the null average was more
        extreme than the observed average across all iterations.

    """ 

    dims = (datasize[0]*datasize[1]*datasize[2], datasize[3])

    # Check for seed input
    if seed == None:
        seed = 1

#     # Make array of zeros for saving true null count across permutation iterations
    perm_output = np.zeros((dims[0]))
    for i in range(n_iter):
        print('> Iteration {} starting...'.format(i))

        # Randomize subject ids and reassign to new groups with same lengths as originals
        fake_subjects = sub_id_dict['left'] + sub_id_dict['right']
        np.random.seed(seed)
        seed += 1 # update seed for next iteration
        np.random.shuffle(fake_subjects)
        
        # Assign randomized ids to new groups
        fake_left = fake_subjects[: len(sub_id_dict['left'])]
        fake_right = fake_subjects[len(sub_id_dict['left']): ]
        assert fake_left != sub_id_dict['left'], "real and fake left group should be different"
        assert fake_right != sub_id_dict['right'], "real and fake right group should be different"

        # Recalculate within-between intersubject using the randomized groups
        perm_inter = Intersubject(data_path, datasize)
        perm_inter.group_isfc({'left': fake_left, 'right': fake_right},
                                compare_method='within_between')

        # Calculate the null distribution average, then compare with real average
        fake_average = perm_inter.isc['within'] - perm_inter.isc['between']
        fake_average = r_average(fake_average, axis=1)

        this_count = tail_null(fake_average, x_avg, tail)
        perm_output = perm_output + this_count

    # Return masked data
    threshold_result = threshold_mask(perm_output, n_iter, sig_level,
                                    tail, output_type, data=x_avg)

    return threshold_result


# # TODO: do this
# # NOTE: this requires doing the entire analysis repeatedly.............
# # Bootstrapping
# def bootstrap(x_dict, sum_dict, n_iter, average_method='median', ones=False, seed=None):

#     data = x_dict['data']
#     vox = 



#     for i in range(n_iter):
#         # get ISC for t

#         # Create matrix of random 1 and -1 
#         np.random.seed(seed + i*100) # explicit random seeds for reproducibility
#         sign_flip = np.random.random(x_matrix.shape) > 0.5
#         sign_flip = np.where(sign_flip == 1, 1, -1) # If true, assign 1; else assign -1

#         # Create null distribution by multiplying voxel averages by 1 or -1; then average across voxels
#         x_permuted = x_matrix * sign_flip
#         fake_average = r_average(x_permuted, method=average_method, axis=1)

#         # Upper one-tailed test for each average voxel
#         # Null hypothesis: True average is not greater than Null average
#         this_count = fake_average > true_average
#         reject_count = reject_count + this_count # add rejection counts per iteration

# def mp_test():

#     with Pool(5)
#!/usr/bin/env python3

# Main intersubject analysis script that reads command-line/shell arguments
# to perform a particular analysis.
# 
# Current Primary options:
# - intersubject method: "entire" or "within_between"
# - nonparametric test:
#   - entire isc: permutation sign-flip
#   - within-between isc: permutation group label

# Add later
# - averaging method
# - ISC computation method (pairwise vs leave-one-out)

# Import functions
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from local_intersubject_pkg.tools import save_data, get_setting
from local_intersubject_pkg.intersubject import Intersubject
from local_intersubject_pkg.nonparametric import perm_signflip, perm_grouplabel

# Hardcoded subject ids (oops)
from subjects import subjects, subs_binge, subs_smoke


def run_intersubject(method, out_type=None, use_fake_data=False, which_fake='range_ids'):
    # do intersubject analysis

    # Get data path
    # NOTE: Currently assumes that simulated data should be used
    # and ignoring use_fake_data positional argument
    data_path = get_setting('input',  which_fake='range_ids')
    datasize = get_setting(which_param='datasize')

    # Create Intersubject object
    intersubject = Intersubject(data_path, datasize)
    
    if method == 'entire':
        intersubject.group_isfc(subjects)

    elif method == 'within_between':
        intersubject.group_isfc({'binge': subs_binge, 'smoke': subs_smoke},
                                compare_method='within_between')

    # Decide whether to only return subset of data
    if out_type == 'isc':
        return intersubject.isc
    elif out_type == 'isfc':
        return intersubject.isfc
    else:
        return intersubject

def run_nonparametric(data, method, iterations, sig_level, data_avg=None, stored_data=None):
    # do nonparametric tests

    # TODO: optionally retrieve data from storage
    if stored_data != None:
        pass

    if method == 'perm_signflip':
        nonparam = perm_signflip(data, iterations, sig_level=sig_level)
    
    # TODO: finish this this morning
    elif method == 'perm_grouplabel':
        data_path = get_setting('input', 'range_ids')
        sub_ids = data.subject_ids
        datasize = data.datasize
        nonparam = perm_grouplabel(data_avg, data_path, sub_ids, datasize,
                iterations, sig_level=sig_level)

    # elif method == 'perm_grouplabel':
    #     nonparam = perm_grouplabel()

    return nonparam

# def run_prep_data():

#     nifti_path = 
#     npy_path
    

def get_analysis_args():

    parser = argparse.ArgumentParser(add_help=True)

    # Create argument groups that cannot be used together.
    inter_group = parser.add_mutually_exclusive_group()
    inter_group.add_argument(
        "-e", "--entire", action="store_true",
        help="""Flag to compute entire intersubject analysis."""
    )
    inter_group.add_argument(
        "-w", "--within_between", action="store_true",
        help="""Flag to compute within-between intersubject analysis"""
    )

    nonparam_group = parser.add_mutually_exclusive_group()
    nonparam_group.add_argument(
        "-s", "--signflip", action="store_true",
        help="""Flag to perform sign-flipping permutation test."""
    )
    nonparam_group.add_argument(
        "-l", "--grouplabel", action="store_true",
        help="""Flag to perform group-label permutation test."""
    )

    parser.add_argument(
        "-a", "--alpha", default=0.05, type=int,
        help="""The significance level to threshold data when computing
        the nonparametric tests."""
    )

    parser.add_argument(
        "-p", "--prep_data", action="store_true",
        help="""Convert Nifti dataset to npy in this projects'
        data directory."""
    )

    parser.add_argument(
        "-z", "--save_data", action="store_true",
        help="""Store results of any computed analysis in 
        output path"""
    )

    # parser.add_argument(
    #     "-i", "--intersub_method", 
    #     choices=['entire', 'within-between'],
    #     help="The intersubject analysis method to compute."
    # )
   
    return parser.parse_args()


# If script is executed/run, then do the stuff below
if __name__ == '__main__':

    # Temporary global vars
    iterations = 20
    significance_level = 0.025

    # Retrieve command line arguments
    args = get_analysis_args()

    # Check intersubject method arguments
    if args.entire:
        intersub_method = 'entire'
    elif args.within_between:
        intersub_method = 'within_between'
    else:
        intersub_method = None

    # Check nonparametric method arguments
    if args.perm_signflip:
        nonparam_method = 'perm_signflip'
    elif args.perm_grouplabel:
        nonparam_method = 'perm_grouplabel'
    else:
        nonparam_method = None


    # Perform the chosen intersubject analysis
    try:
        if intersub_method != None:
            intersubject = run_intersubject(intersub_method)
        print(f"...'{intersub_method}' intersubject method performed successfully.")
    except:
        print(f"...'{intersub_method}' intersubject method failed to complete :'(")

    # Do a nonparametric test
    if nonparam_method != None:
        if nonparam_method == 'perm_signflip':
            results = perm_signflip(intersubject['entire'], iterations, sig_level=significance_level)

    # Save results
    # - arrays
    # - figures
    if args.save_data:

        # NOTE: generic filename for now
        file_ = 'results.npy' 
        output_path = get_setting('output')
        filepath = os.path.join(output_path, file_)
        save_data(filepath, results)
        
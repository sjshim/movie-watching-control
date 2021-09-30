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
import seaborn as sns

from local_intersubject_pkg.tools import save_data, get_setting, prep_data
from local_intersubject_pkg.intersubject import Intersubject
from local_intersubject_pkg.nonparametric import perm_signflip, perm_grouplabel

# Hardcoded subject ids (oops)
from subjects import subjects, subs_binge, subs_smoke


def choose_intersubject(method, out_type=None, use_fake_data=False, 
                        which_fake='range_ids'):
    # do intersubject analysis

    # Get data path
    # NOTE: Currently assumes that simulated data should be used
    # and ignoring use_fake_data positional argument
    if use_fake_data:
        data_path = get_setting('input',  which_fake=which_fake)
    else:
        data_path = get_setting('input', which_input='npy')
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

def choose_nonparametric(data, method, iterations, sig_level, data_avg=None, 
                        stored_data=None):
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

    return nonparam

def save_visualization(result, output_path, *method_names):

    # NOTE: currently, should very simply produce a heatmap for ISC. 
    # TODO: in the future, this script should also
    #   - save figures for ISFC
    #   - be useable for all potential figures that this analysis will need;
    #       even for IS-RSA potentially. It will need to be decided in the 
    #       future whether this script handles fmri, behavioral, and 
    #       fmri x behavioral (IS-RSA) analyses
    #   - plotting brain projections

    # =====
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(result, center=0, vmin=-1, vmax=1, ax=ax)
    plt.title(" ".join(method_names) + "heatmap")
    file_name = "_".join(method_names) + "results_heatmap.png"
    plt.savefig(os.path.join(output_path, file_name))

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
        the nonparametric tests. Default value is 0.05"""
    )

    parser.add_argument(
        "-i", "--iterations", default=100, type=int,
        help="""The number of iterations to be performed by the isc permutation
        tests. Default value is 100."""
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

    parser.add_argument(
        "-v", "--visualize", action="store_true",
        help="""Visualize results. NOTE: beta feature, very inflexible atm."""
    )

    # parser.add_argument(
    #     "-i", "--intersub_method", 
    #     choices=['entire', 'within-between'],
    #     help="The intersubject analysis method to compute."
    # )
   
    return parser.parse_args()

def main():
    # Retrieve command line arguments
    args = get_analysis_args()

    # Optionally prep data
    if args.prep_data:
        prep_data(get_setting('nifti'))

    # Check intersubject method arguments
    if args.entire:
        intersub_method = 'entire'
    elif args.within_between:
        intersub_method = 'within_between'
    else:
        intersub_method = None

    # Check nonparametric method arguments
    if args.signflip:
        nonparam_method = 'perm_signflip'
    elif args.grouplabel:
        nonparam_method = 'perm_grouplabel'
    else:
        nonparam_method = None

    # Error if both method arguments are None
    assert intersub_method != nonparam_method, 'Provide at least one method argument'
    
    # ========================================
    # Perform the chosen intersubject analysis
    try:
        if intersub_method != None:
            intersub_results = choose_intersubject(intersub_method)
            print(f"...'{intersub_method}' intersubject method performed successfully.")
    except:
        print(f"...'{intersub_method}' intersubject method failed to complete :'(")

    # ==========================
    # Perform nonparametric test
    try: 
        if nonparam_method != None:
            nonparam_results = choose_nonparametric(intersub_results, 
                                nonparam_method, args.iterations, args.alpha)
            print(f"...'{nonparam_method}' nonparametric test performed successfully.")
    except:
        print(f"...'{nonparam_method}' nonparametric test failed to complete.")

    # NOTE: currently inflexible way of dealing with two optional output types
    if nonparam_method is not None:
        final_results = nonparam_results
    elif intersub_method is not None:
        final_results = intersub_method

    # NOTE: this is currently inflexible to selectively saving results for ISC
    # on its own without permutation test thresholding; I might want to 
    # change this in the future
    #
    # Save results
    # - optional isc and/or thresholded results
    # - arrays
    # - figures
    # NOTE: currently assumes that both interub and nonparam analyses have been performed
    if args.save_data:
        results_file = f"{intersub_method}_isc_{nonparam_method}_results.npy" 
        output_path = get_setting('output')
        filepath = os.path.join(output_path, results_file)
        save_data(filepath, final_results)
    if args.visualize:
        save_visualization(final_results, output_path, intersub_method, nonparam_method)
# If script is executed/run, then do the stuff below
if __name__ == '__main__':
    main()
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
from nibabel import Nifti1Image

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting

from local_intersubject_pkg.tools import save_data, get_setting, prep_data
from local_intersubject_pkg.intersubject import Intersubject
from local_intersubject_pkg.nonparametric import perm_signflip, perm_grouplabel

# Hardcoded subject ids (oops)
# from subjects import subjects, subs_binge, subs_smoke


def choose_intersubject(method, use_fake_data=None):
    # Choose intersubject correlation method to compute

    # Get data path
    # NOTE: Currently assumes that simulated data should be used
    # and ignoring use_fake_data positional argument
    if use_fake_data != None:
        data_path = get_setting(get_fake=use_fake_data)
    else:
        data_path = get_setting(get_path='npy_path')
    datasize = get_setting(get_param='datasize')

    if method == 'entire':
        subs = get_setting(get_ids='all')
    elif method == 'within_between':
        subs = get_setting(get_param='sub_ids')
        subs.pop('all', None) # remove 'all' key, assumes two group keys remain

    # Calculate ISC with specified group comparison method
    intersubject = Intersubject('isc', method, os.path.join(data_path, 'sub-{}_func_small.npz'), datasize)
    intersubject.compute(subs)
    return intersubject.result

def choose_nonparametric(method, iterations, sig_level, data=None, 
                        stored_data=None):
    # do nonparametric tests

    # TODO: optionally retrieve data from storage
    if stored_data != None:
        pass

    if method == 'perm_signflip':
        nonparam = perm_signflip(data, iterations, sig_level=sig_level)
    elif method == 'perm_grouplabel':
        data_path = get_setting(get_path='npy_path') # Uses preprocessed npy data for recalculating ISC
        # Get info from Intersubject object attributes
        sub_ids = data.subject_ids
        datasize = data.datasize
        nonparam = perm_grouplabel(data, data_path, sub_ids, datasize,
                iterations, sig_level=sig_level)

    return nonparam

def save_visualization(result, output_path, *method_names):
    # TODO: in the future, this script should also
    #   - save figures for ISFC
    #   - be useable for all potential figures that this analysis will need;
    #       even for IS-RSA potentially. It will need to be decided in the 
    #       future whether this script handles fmri, behavioral, and 
    #       fmri x behavioral (IS-RSA) analyses
    #   - plotting brain projections

    # =====
    try:
        # fig, ax = plt.subplots(figsize=(12, 7))
        # sns.heatmap(result, center=0, vmin=-1, vmax=1, ax=ax)
        # plt.title(" ".join(method_names) + " heatmap")
        ds = get_setting(get_param='datasize')
        affine = get_setting(get_param='affine')
        result_niimg = Nifti1Image(
            result.reshape(ds[0], ds[1], ds[2]),
            affine)
        file_name = "_".join(method_names) + "_results_brain_plot"
        filepath = os.path.join(output_path, file_name)
        # plt.savefig()
        title = " ".join(method_names) + " brain plot"

        # Save brain plot as static png file
        plotting.plot_stat_map(result_niimg, threshold=None, title=title, 
                output_file=filepath + '.png')
        # Save brain plot as interactive html
        plotting.view_img(result_niimg, threshold=None,  title=title,
                output_file=filepath + '.html')
        print(f"...successfully saved visualization at:\n{filepath}\n")
    except:
        print(f"...failed to save visualization at:\n{filepath}\n")

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
    print(f"Running {__file__}...\n")
    # Retrieve command line arguments
    args = get_analysis_args()

    # Optionally prep data
    if args.prep_data:
        print("Running prep_data...")
        try:
            prep_data()
            print("...successfully prepped data.\n")
        except:
            print("...failed to prep data.\n")


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
    analysis_params = ", ".join([str(i) for i in [intersub_method, nonparam_method, 
                        args.alpha, args.iterations] if i is not None])
    print(f"Analysis parameters:\n>   {analysis_params}\n")

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
            # NOTE: currently assumes only ISC is provided, not ISFC
            nonparam_results = choose_nonparametric(nonparam_method, 
                    args.iterations, args.alpha, data=intersub_results.isc)
            print(f"...'{nonparam_method}' nonparametric test performed successfully.")
    except:
        print(f"...'{nonparam_method}' nonparametric test failed to complete.")

    # NOTE: currently inflexible way of dealing with two optional output types
    if nonparam_method is not None:
        final_results = nonparam_results
    elif intersub_method is not None:
        final_results = intersub_results

    # NOTE: this is currently inflexible to selectively saving results for ISC
    # on its own without permutation test thresholding; I might want to 
    # change this in the future
    #
    # Save results
    # - optional isc and/or thresholded results
    # - arrays
    # - figures
    # NOTE: currently assumes that both interub and nonparam analyses have been performed
    output_path = get_setting(get_path='data_output')
    if args.save_data:
        try:
            results_file = os.path.join(output_path, 
                    f"{intersub_method}_isc_{nonparam_method}_results.nii.gz")
            save_data(results_file, data=final_results,
                    affine=get_setting(get_param='affine'),
                    datasize=get_setting(get_param='datasize'))
            print(f"...successfully saved final results at:\n{results_file}\n")
        except:
            print(f"...failed to save final results at:\n{results_file}\n")

    if args.visualize:
        save_visualization(final_results, output_path, intersub_method, nonparam_method)
# If script is executed/run, then do the stuff below
if __name__ == '__main__':
    main()
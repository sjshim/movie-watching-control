#!/usr/bn/env python3
# run_analysis.py
# Author: Jason Zamorano
# Purpose: Script to perform 'intersubject' analyses.

from functools import partial
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, plot_glass_brain

from local_intersubject_pkg.intersubject import (isc, wmb_isc, finn_isrsa,
                                        isc_by_segment, movie_seg_compute, 
                                        tr_mask_from_segments, timestamps_from_segments)
from local_intersubject_pkg.nonparametric import (perm_signflip, perm_grouplabel, 
                                        perm_mantel)
from local_intersubject_pkg.utils.tools import get_files_dict, get_setting
from local_intersubject_pkg.utils.plot_utils import transform_and_plot

MAIN_STATS = {
    'loo-isc': isc,
    'pairwise-isc': partial(isc, pairwise=True),
    'wmb-isc': wmb_isc,
    'finn-isrsa': finn_isrsa
}

STAT_TESTS = {
    'loo-isc': {'perm_signflip': perm_signflip},
    'pairwise-isc': {'perm_signflip': perm_signflip},
    'wmb-isc': {'perm_grouplabel': partial(perm_grouplabel, stat_func=wmb_isc)},
    'finn-isrsa': {'perm_mantel': perm_mantel}
}

PLOT_FUNCS = {
    'stat-map': plot_stat_map,
    'glass-brain': plot_glass_brain
}


def load_data(cli_args, masker):
    """Load data from filenames using MultiNiftiMasker, then return data as
    numpy arrays."""
    custom_transform = lambda func_dict : np.swapaxes(np.array(
                                        masker.transform(func_dict.values())), 
                                        axis1=1, axis2=2).T

    # Get each group's filepaths separately
    sub_ids = get_setting(get_param='sub_ids')
    func_dict = {}
    for group in cli_args.group:
        func_dict[group] = get_files_dict(get_setting(get_path='nifti_path'), 
                            sub_ids[group]).values()

    # Fit masker to all groups, but transform group data separately
    masker.fit([file for group in func_dict for file in func_dict[group]])
    data_dict = {group: custom_transform(func_dict[group]) for group in func_dict}
    return data_dict


def compute_analysis(cli_args, data):
    """Compute main intersubject functions and/or statistical tests."""
    # Set up multimasker instance
    masker_args = dict(n_jobs=-1)
    if cli_args.mask: # might move this arg to script_settings.json
        masker_args['mask_img'] = cli_args.mask
    else:
        masker_args['mask_strategy'] = 'gm-template'
    masker = MultiNiftiMasker(**masker_args)

    data_dict = load_data(cli_args, masker)

    if cli_args.main_stat:
        main_func = MAIN_STATS[cli_args.main_stat]
        unthresholded_results = main_func(*[data_dict.values()])
    if cli_args.stat_test:
        test_args = dict(n_iter=cli_args.n_iterations)
        test_func = STAT_TESTS[cli_args.stat_test]
        thresholded_results = test_func(unthresholded_results, **test_args)
    
    visualize_results(cli_args, masker, thresholded_results)


def visualize_results(cli_args, masker, data):
    """Plot results and save to disk."""
    # temporary plotting configuration
    plot_name = '_'.join([cli_args.main_stat, cli_args.stat_test, 'stat-map'])
    custom_stat_map = transform_and_plot(masker.inverse_transform, PLOT_FUNCS['stat-map'])
    custom_stat_map(data, 
                output_file=Path().cwd()/(plot_name+'.png'))

 
def get_cli_args():
    """Get command line arguments for this script."""
    parser = ArgumentParser(add_help=True)
    parser.add_argument('-m', '--main-stat', type=str,
        help="""The intersubject analysis function to use.""")
    parser.add_argument('-t', '--stat-test', type=str,
        help="""The statistical test to use on unthresholded statistical 
        results.""")
    parser.add_argument('-g', '--group', 
        type=str, nargs='+', # accept at least one argument
        choices=[*get_setting(get_param='sub_ids').keys()],
        help="""Choose which group you want to perform analysis on. Choices
        must either be 'all', or the specific name of the subject group stored in 
        script_settings.json['Parameters']['sub_ids']""")
    parser.add_argument('-r', '--critical-region', 
        type=float, default=0.05,
        help="""(Optional) Specify the critical region that the statistical test
        will use to threhsold data with. Default value is 0.05""")
    parser.add_argument('-i', '--n-iterations', 
        type=int, default=1000,
        help="""(Optional) Number of iterations to use in nonparametric tests.""")
    parser.add_argument('-s', '--save-data',
        type=str, choices=['main', 'test', 'all'], default='all',
        help="""(Optional) Choose whether to save analysis results to disk.
        
        Possible options include 'main', 'test', or 'all';
        'all' is chosen by default.""")
    
    return parser.parse_args()


def main():
    print(f"Running {__file__}...\n")
    cli_args = get_cli_args()
    assert all(v is None for v in [cli_args.main_stat, cli_args.stat_test]) == False, "Option must be provided for either --main-stat or --test arguments"
    compute_analysis(cli_args)


if __name__ == "__main__":
    main()
#!/usr/bn/env python3
# new_main.py
# Purpose: Script to perform all intersubject-like analyses.

from functools import partial
from argparse import ArgumentParser

from local_intersubject_pkg.intersubject import (isc, wmb_isc, isc_by_segment,
                                        movie_seg_compute, tr_mask_from_segments,
                                        timestamps_from_segments, finn_isrsa)
from local_intersubject_pkg.nonparametric import (perm_signflip, perm_grouplabel, 
                                        perm_mantel)

MAIN_STATS = {
    'loo-isc': isc,
    'pairwise-isc': partial(isc, pairwise=True),
    'wmb-isc': wmb_isc,
    'finn-isrsa': finn_isrsa
}

STAT_TESTS = {
    'loo-isc': {'perm_signflip': perm_signflip},
    'pairwise-isc': {'perm_signflip': perm_signflip},
    'wmb-isc': {'perm_grouplabel': perm_grouplabel},
    'finn-isrsa': {'perm_mantel': perm_mantel}
}


def load_data(a):
    """Load data from filenames using NiftiMasker, then return data as
    numpy arrays."""
    pass


def compute_analysis():
    """Compute main intersubject functions and/or statistical tests."""
    pass


def visualize_results():
    """Plot results and save to disk."""
    pass

 
def get_cli_args():
    """Get command line arguments for this script."""
    parser = ArgumentParser(add_help=True)
    parser.add_argument('-m', '--main-stat', type=str,
        help="""The intersubject analysis function to use.""")
    parser.add_argument('-t', '--test', type=str,
        help="""The statistical test to use on unthresholded statistical 
        results.""")
    parser.add_argument('-r', '--critical-region', 
        type=float, default=0.05,
        help="""Specify the critical region that the statistical test
        will use to threhsold data with. Default value is 0.05""")
    parser.add_argument('-i', '--iterations', 
        type=int, default=1000,
        help="""Number of iterations to use in nonparametric tests.""")
    parser.add_argument('-s', '--save-data',
        type=str, choices=['main', 'test', 'all'], default='all',
        help="""Choose whether to save analysis results to disk.
        
        Possible options include 'main', 'test', or 'all';
        'all' is chosen by default.""")
    
    return parser.parse_args()


def main():
    print(f"Running {__file__}...\n")
    cli_args = get_cli_args()

    if cli_args.main_stat:
        pass



if __name__ == "__main__":
    main()
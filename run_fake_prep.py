#!/usr/bin/env python3

# run_fake_prep: This script creates fake subject data used for local testing
# of both the intersubject package and scripts for doing the analyses 
# automatically. The data is 4d with the shape (4, 4, 4, 10).
# 
# The files are created either using lists of real subject ID's or by using
# the desired number/range of subjects (from which their IDs will be derived).
#  
# The resulting fake data will be store as Numpy ".npy" files and located in the
# directory "data/inputs/fake_data/" and "real_ids/" or "range_ids/" depending
# on which type was chosen. 

import os
import sys

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# import custom funcs
import local_intersubject_pkg.tools as tools
from subjects import subjects, subs_binge, subs_smoke

def setup_fake_data():

    tools.create_fake_data(path_dict['fake_ids']+file_fake, id_list = subjects)
    tools.create_fake_data(path_dict['fake_range']+file_fake, no_of_subjects = sub_range)

    # check if files are present
    import glob
    for id_ in [i*2+1 for i in range(sub_range)]:
        for path in glob.iglob(path_dict['fake_range']+'/sub-{}.npy'.format(id_), recursive=True):
            print(path)
            if path == path_dict['fake_range']+'/sub-{}.npy'.format(id_):
                print('yuh')
            else:
                print('damn')

    # Get data id:filepath dicts, then save filtered data
    ids_dict = tools.get_files_dict(path_dict['fake_ids'] + file_fake, subjects)
    range_dict = tools.get_files_dict(path_dict['fake_range'] + file_fake, [i*2+1 for i in range(sub_range)])

    # NOTE: make cutoff mean=0; it makes the data far too sparse for testing purpose (and might not even be useful
    # for IRL analysis). Mean cutoff was once originally 3000, a carryover from YC
    cutoffs = {'col': 10, 'mean': 0}
    tools.prep_data(ids_dict, path_dict['filter_ids'] + file_filter, cutoffs['col'], cutoffs['mean'])
    tools.prep_data(range_dict, path_dict['filter_range'] + file_filter, cutoffs['col'], cutoffs['mean'])
    
    return ids_dict, range_dict


if __name__ == "__main__":

    args = sys.argv

    # Define data input paths strings
    cwd = os.getcwd()
    path_dict = {
        'fake_ids': cwd + "/data/inputs/fake_ids",
        'fake_range': cwd + "/data/inputs/fake_range",
        'filter_ids': cwd + "/data/filter/fake_ids",
        'filter_range': cwd + "/data/filter/fake_range"
    }

    # Save filename templates
    file_fake = '/sub-{}.npy'
    file_filter = '/filter_sub-{}.npz'

    # Create each directory
    for path in path_dict:
        tools.create_directory(path_dict[path])

    sub_range = 15

    # Create fake data with using real and range IDs
    ids_dict, range_dict = setup_fake_data()


    
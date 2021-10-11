# tools.py: A module containing useful non-statistical functions
#  

import os
import glob
import json
from json.decoder import JSONDecodeError

import numpy as np
import nibabel as nib

# =============================
# Basic data handling functions
# =============================

# Make directories, with error exception
def create_directory(dir_):
    """
    Creates directory if it doesn't exist already and prints directory
    creation errors to the console.
    """

    # TODO: replace print statements with logging
    # Make these directories; print results or errors
    if os.path.exists(dir_) == False:
        print(f'> Path: {dir_}\n...could not be found.\n> Attempting to create directory...\n')
        try:
            os.makedirs(dir_)
            print(f'> Path: {dir_}\n...successfully created.\n')
        except:
            print(f'> Failed to create path: {dir_}\n')
    else:
        print(f'> Path: {dir_}\n...already exists.\n')


def save_data(dir_, *args, **kwargs):
    """
    Uses numpy's save(), savez() or savez_compressed(), 
    but double-checks if directory exists too
    """
    # assert os.path.exists(dir_), f"Path {dir_} does not exists."

    try:
        if dir_.endswith('.npy'):
            np.save(dir_, *args)

        elif dir_.endswith('.npz'):
            np.savez(dir_, **kwargs)

        # Log success here
        assert os.path.exists(dir_), "Path did not work"
        print(f"> Data file at '{dir_}' successfully saved.\n")
    except:
        # Log failure here
        print(f"> Data file at '{dir_}' failed to save.\n")

# ============================
# Pre-analysis setup functions
# ============================

# Get dict of each subject id and associated filepath
def get_files_dict(path, id_list):
    """
    Get dict of each subject id and its associated filepath. The supplied filepath
    should contain curly brackets for string formatting with subject ids; it can
    contain glob patterns where necessary.

    Intended for use during data preparation, before analysis.
    """
    id_file_dict = {}
    for id_ in id_list:
        try:
            glob_pattern = path.format(id_)
            for file in glob.iglob(glob_pattern, recursive=True):
                id_file_dict[id_] = file
        except:
            print(f"Failed to retrieve filepath for subject {id_} using glob pattern...\n{glob_pattern}")
    return id_file_dict

def check_datasizes(file_dict, check_3d_equality=True, return_4d_tuple=False, return_sub_id=False):
    # Check data dimensions and shortest TR length
    datasizes = {}
    for sub in file_dict:
        # NOTE: the .shape attr was chosen based on documentation and to make code more consise; 
        # this has not been tested yet and may not work as intended (7/16/21)
        datasizes[sub] = nib.load(file_dict[sub]).shape

    # Return error if 3d volume dimensions are not equal between subjects i and i-1
    if check_3d_equality:
        last_ds = {}
        for this_sub in datasizes:
            if not last_ds:
                last_ds[this_sub] = datasizes[this_sub]
            else:
                last_sub = list(last_ds.keys())[0]
                this_ds = datasizes[this_sub]
                # BUG: this line causes unhashable error because last_ds is a dict, not a tuple like expected. Is the last_ds dict even necessary?
                # (8/18/21) Yes, because I created the option of returning lowest-TR subject's entire datasize, so it is conveninent to use their sub as a key in a dict
                assert last_ds[last_sub][:3] == this_ds[:3], \
                    f"All subjects must have same 3d shape; previous sub {last_sub} had \
                        {last_ds[last_sub]} while current sub {this_sub} had {this_ds}"
                del last_ds[last_sub]
                last_ds[this_sub] = this_ds

    # TODO: optionally return subject-id alongside their data dimensions
    # Return lowest TR integer or 4d tuple containing lowest TR (4th dim)
    min_tr = min([datasizes[i][3] for i in datasizes])
    # NOTE: this optional tuple return currently depends on last_ds from the check_3d_equality arg; ensure whether this dependency is necessary
    if return_4d_tuple:
        return (*last_ds[sub][:3], min_tr)
    else:
        return min_tr

# Reshape 4d fmri data into 2d and save output as .npy filetype
# NOTE: Should I separate the following into funcitons?
# get_files_dict, nifti_to_npy, get_datasizes, all under parent func prep_data?
# I feel that this way, I can definitively feed prep_data only settings_file.json info
def prep_data(files_dict=None, nifti_path=None, output_path=None, 
            cutoff_mean=None, out_file="sub-{}_func_small.npz"):
    """
    Reshapes 4d fmri data into a 2d numpy array and saves the output as a
    .npy file. Input data can be either Nifti format (.nii or .nii.gz) or 
    numpy array.

    Parameters
    ----------
    files_dict : dict
        Dict where key is a subject id and value is the subject's specific nifti filepath

    """
    # notes: outline of the steps
    # - include filepath dict function somewhere?

    # 1) Retrieve settings settings file NIFTI path
    # ?) get files dict
    # ?) get subjects list from .json settings file
    # 2) check TR if true, save results to settings file parameters
    # 3) convert NIFTI to npy if true, save files to settings file npy path default
    # or user-defined path

    # Setup nifti and output paths
    try:
        if nifti_path is None:
            # nifti_path from script_settings.json is assumed to include have 
            # brackets for str formatting
            nifti_path = get_setting('input', 'nifti')
        assert nifti_path != '', "Paths/nifti_path must be defined in script_settings.json"
        if output_path is None:
            output_path = get_setting('input', 'npy')
        assert output_path != '', "Paths/npy_path must be defined in script_settings.json"

        # Get files dict
        if files_dict is None:
            try:
                sub_ids = get_setting(which_param='sub_ids', which_ids='all')
                assert (not sub_ids) is False, "['Parameters']['sub_ids']['all'] in script_settings.json cannot be empty"
                files_dict = get_files_dict(nifti_path, sub_ids)
                assert sub_ids.copy().sort() == list(files_dict.keys()).copy().sort(), "Sub ids and generated filepath dict keys did not match for some reason"
            except:
                print(f"Couldn't obtain files dict from path:\n{nifti_path}\n...for subject ids:\n{sub_ids}")
    except Exception:
        print("Couldn't retrieve nifti_path or npy_path from script_settings.json")

    # Check for datasize and minimum-TR across all subjects' time-series
    datasize = check_datasizes(files_dict, return_4d_tuple=True)
    cutoff_column = datasize[3] # get lowest TR from nifti files

    # NOTE: this is ugly and revisions should make get_settings more flexible
    # for either retrieving from or saving settings to script_settings.json
    settings_file = "script_settings.json"
    with open(settings_file, 'r+') as file_:
        config = json.load(file_)
        config['Parameters']['datasize'] = datasize
        json.dump(config, file_, indent=4)

    # Process each subjects' data
    for id_ in files_dict:
        id_path = files_dict[id_]
        # Check for numpy.npy or NIfTI file
        try:
            if id_path.endswith(".npy"):
                data = np.load(id_path) # .npy files are assumed to have same dimensions
            elif id_path.endswith(".gz"):
                data = nib.load(id_path).get_fdata()[:,:,:, 0: cutoff_column]
        except:
            print("Unrecognized file type.")
            break

        # reshape 4d array to matrix
        # Note: rows=voxels, columns=TRs
        datasize = data.shape # NOTE: this is the second datasize retrieval and could cause intended errors 
        data = data.reshape((datasize[0] * datasize[1] * datasize[2]),
                            datasize[3]).astype(np.float32)
        
        # create boolean mask based on each voxel's mean (optional)
        if cutoff_mean is None:
            cutoff_mean = np.min(data, axis=1)
        mask = np.mean(data, axis=1) > cutoff_mean
        
        # filter voxels using mask
        data = data[mask, :]
        
        # save filtered data to numpy.npz file
        save_data(os.path.join(output_path, out_file.format(id_)),
                data=data, mask=mask)

def create_fake_data(output_path, datasize=(4, 4, 4, 10), no_of_subjects=None, id_list=None):
    """
    Create fake fmri data saved as numpy.npy files. Data is created
    using either a list of subject IDs or a number of 
    subjects.
    
    Note:
    - 
    - Default datasize is 4,4,4,10 or 64 voxels x 10 TRs
    """
    
    if id_list == None:
        # Use range to create ids
        subjects = [i*2+1 for i in range(no_of_subjects)]
    else:
        subjects = id_list
        
    # create fake data then save
    for i, sub in enumerate(subjects):
        seed = i + 500 # hacky way of getting random seed per loop
        np.random.seed(seed)
        fake_data = np.random.randint(1000, 5000, size=datasize)
        save_data(output_path.format(sub), fake_data)

def get_setting(in_or_out=None, which_input=None, which_fake=None, 
                which_param=None, which_ids=None):
    """
    Get details from the script settings JSON file created for this analysis.
    """
    # Get datapaths
    settings_file = "script_settings.json"
    try:
        with open(settings_file) as file_:
            config = json.load(file_)

            # Check for parameter request
            if which_param != None: 
                output = config['Parameters'][which_param]
                if which_ids != None:
                    output = output[which_ids]

            # Check for filepath request
            elif in_or_out == 'input':

                # Check for real data
                if which_input == 'npy':
                    output = config['Paths']['npy_path']
                elif which_input == 'nifti':
                    output = config['Paths']['nifti_path']

                # Check for 
                elif which_fake == 'range_ids':
                    output = config['Fake Data Paths']['filter_range']
                elif which_fake == 'real_ids':
                    output = config['Fake Data Paths']['filter_real']
            
            elif in_or_out == 'output':
                output = config['Paths']['data_outputs']

            # NOTE: this has been causing issues, and is incompatible with
            # the stored nifti glob path; this may be removed in the future
            # # Double check that directory exists
            # assert os.path.exists(output), f"The path...\n{output}\n...could not be found or does not exist."
    except JSONDecodeError:
        args = ", ".join([i for i in [which_param, which_ids, in_or_out, \
                            which_input, which_fake] if i is not None])
        print(f"...Failed to retrieve setting '{args}' from settings file {settings_file}")

    # Return result
    return output
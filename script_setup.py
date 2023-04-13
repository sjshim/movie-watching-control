#!/usr/bin/env python3
# script_setup.py: Create `script_settings.json` containing data i/o directories
# and other analysis parameters.
import os
import sys
import argparse
# import configparser
import json
import glob

from local_intersubject_pkg.utils.tools import create_directory

def create_script_settings(project_path, data_dest, bids_path=None, 
                        preproc_path=None, create_dir=False, 
                        sub_ids_json=None, settings_file=None):
    """
    Create a configuration file containing filepaths, parameters, and other
    persistent info that are used by this analysis. The resulting file is
    a JSON file that can also be changed manually.

    Its filepaths represent the default directory structure of this project. 
    """
    # ========================================================
    # Create dicts of filepaths
    paths = {
        "settings_path": settings_file,
        "project_path": project_path, # incase it's not cwd for some reason

        "BIDS_path": bids_path,
        "preproc_func": preproc_path,

        "data_dest": data_dest,
        "data_input": os.path.join(data_dest, "data", "input"),
        "data_output": os.path.join(data_dest, "data", "output")
    }
    # Use os.path.join for OS compatible filepaths
    paths['npy_path'] = os.path.join(paths['data_input'], 'npy_data')

    # Setup parameters needed between scripts
    parameters = {
        "datasize" : (),
        "affine": (),
        "sub_ids": {"all":[]} # default dict if a subject id json isn't provided
    }
    # Retrieve subject id lists from a separate .json file if provided
    if sub_ids_json is not None:
        try:
            with open(sub_ids_json) as file_:
                sub_ids_dict = json.load(file_)
                assert "all" in list(sub_ids_dict.keys()), "'all' key must exist within the provided subject id json file."
                for label in sub_ids_dict:
                    parameters['sub_ids'][label] = sub_ids_dict[label]
        except:
            print(f"Failed to retrieve and save subject id lists from the provided file {sub_ids_json}.")
    # Otherwise, use the default subject id list
    else:
        subjects = glob.glob(bids_path + '/*')
        subjects = sorted([os.path.basename(subject_path) for subject_path in subjects])
        parameters['sub_ids']['all'] = subjects
    # Setup paths for fake test data
    fake_paths = {
        "real_ids": ["fake_data", "real_ids"],
        "range_ids": ["fake_data", "range_ids"],
        "filter_real": ["fake_data", "filter", "filter_real"],
        "filter_range": ["fake_data", "filter", "filter_range"]
    }

    for i in fake_paths:
        fake_paths[i] = os.path.join(paths['data_input'], *fake_paths[i])

    # =========================
    # Save all script settings as a .json file
    main_sections = {
        'Paths': paths, 
        'Parameters': parameters, 
        'Fake Data Paths': fake_paths}
    # main_subsections = [paths, parameters, fake_paths]
    
    try:
        with open(settings_file, 'w') as outfile:
            json.dump(main_sections, outfile, indent=4)
        assert os.path.exists(settings_file)
        print(f"...Successfully wrote '{settings_file}' configuration file")            
    except:
        print(f"...Could not write '{settings_file}' configuration file") 

    # ==========================
    # Create processed data I/O directories if requested (excluding nifti path)
    if create_dir:
        try:
            print(f"\nCreating directories...")
            for path in ['data_input', 'data_output', 'npy_path']:
                create_directory(paths[path])
            for path in fake_paths:
                create_directory(fake_paths[path])
            print(f"...succesfully created directories.\n")
        except:
            print(f"...failed to create directories.\n")

def get_setup_arguments():
    # fancy command line arguments
    parser = argparse.ArgumentParser(
        add_help=True,
        description="""This script stores the relevant filepaths and parameters 
        used by this analysis to script_settings.json. This allows those
        details to be human readable and editable (when running this script or
        even after script_settings.json has been created), and accessible by 
        other parts of code (both when performed interactively or by script);
        the result is hopefully more reproducible and transparent analysis 
        pipeline. Details that are not provided to this script will be empty
        strings by default in most cases and it is up to the user to manually 
        provide them to script_setting.json afterward."""
    )
    parser.add_argument(
        "-s", "--sub_ids_json", default=None,
        help="""Optional. Specify the name of a .json file containing a dict
            where keys are group/condition labels and the subject id lists are
            values. A key named 'all' containing all subject ids in the dataset
            is required."""
    )
    parser.add_argument(
        "-b", "--bids_path", default="",
        help="""The root path containing subjects' functional and anatomical
        Nifti data. If the bids_path arg is not provided, then it will be 
        inferred from the full paths from the nifti_func and nifti_anat args; if 
        both nifti_func and nifti_anat are not provided, then bids_path must
        be provided manually within script_settings.json after it has been 
        created by this script."""
    )
    parser.add_argument(
        "-f", "--preproc_path", default="",
        help="""The path for subjects' preprocessed data directory. preproc_path
        can either be the absolute or relative path;
        relative paths requires that the absolute BIDS parent path is supplied
        to either the bids_path arg or manually provided to
        script_settings.json"""
    )
    parser.add_argument(
        "-p", "--project_path", default=None,
        help="""Optional; The location of this project. If not provided, 
        then it is defined as the current working directory of this script."""
    )
    parser.add_argument(
        "-d", "--data_dest", default=None,
        help="""Optional; The parent data directory destination to accomadate a different
        location from the project directory containing the code. If not provided, then data_dest
        is made the same as project_path."""
    )
    parser.add_argument(
        "-c", "--create_dir",
        help="""Optional; Create directory structure as defined in this settings file. If this flag
        is not provided, then directory creation is supressed.""",
        action="store_true"
    )
    parser.add_argument(
        "-x", "--settings_fn", default='script_settings.json',
        help="""Optional; Define a different filename for your script settings
        json file. Otherwise, the default name is 'script_settings.json'."""
    )

    return parser.parse_args()

def main():
    # Get arguments from command line
    args = get_setup_arguments()
    project_path = args.project_path

    if project_path is None:
        project_path = os.getcwd()
    if args.data_dest is None:
        data_dest = project_path
    else:
        data_dest = args.data_dest

    # TODO: handle os.path incompatible user paths or replace with Pathlib.
    # # add starting dash if missing
    # if nifti_path[0] != '/':
    #     nifti_path = '/' + nifti_path

    print(f"Running script_setup.py, creating settings json file...\n")
    if args.sub_ids_json is not None:
        print(f"Retrieving subject ids from {args.sub_ids_json}...")
    print(f'Making json settings file...\n')
    print(f"BIDS path: \n--> '{args.bids_path}'")
    print(f"Preproc path: \n--> '{args.preproc_path}'")
    print(f"Project path: \n--> '{project_path}'")
    print(f"Data destination: \n--> '{data_dest}'")
    print("\nPlease confirm that directories above are correct before proceeding with your analyses!\n")

    # Make the settings file!
    create_script_settings(project_path=project_path, data_dest=data_dest, 
                        bids_path=args.bids_path, preproc_path = args.preproc_path,
                        create_dir=args.create_dir, sub_ids_json=args.sub_ids_json, 
                        settings_file=args.settings_fn)

if __name__ == "__main__":
    main()
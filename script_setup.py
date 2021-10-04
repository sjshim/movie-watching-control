#!/usr/bin/env python3
# script_setup.py: Create `script_settings.json` containing data i/o directories
# and other analysis parameters.
import os
import sys
import argparse
# import configparser
import json

from local_intersubject_pkg.tools import create_directory

def create_script_settings(nifti_path, project_path, data_dest, 
                            create_dir=False, sub_ids_json=None):
    """
    Create a configuration file containing filepaths, parameters, and other
    persistent info that are used by this analysis. The resulting file is
    a JSON file that can also be changed manually.

    Its filepaths represent the default directory structure of this project. 
    """
    # ========================================================
    # Create dicts of filepaths
    settings_file = "script_settings.json" # script settings file name
    paths = {
        "settings_path": settings_file,
        "project_path": project_path, # incase it's not cwd for some reason

        "data_dest": data_dest,
        "data_inputs": os.path.join(data_dest, "data", "input"),
        "data_outputs": os.path.join(data_dest, "data", "output"),

        "nifti_path": nifti_path
    }
    # Use os.path.join for OS compatible filepaths
    paths['npy_path'] = os.path.join(paths['data_inputs'], 'npy_data')

    # Setup parameters needed between scripts
    parameters = {
        "datasize" : (),
        "sub_ids": {"all":[]} # default dict if a subject id json isn't provided
    }
    # Retrieve subject id lists from a separate .json file if provided
    if sub_ids_json is not None:
        try:
            with open(sub_ids_json) as file_:
                sub_ids_dict = json.load(file_)
                assert "all" in list(sub_ids_dict.keys()), "'all' key must exist within the provided subject id json file."
                for label in sub_ids_json.keys():
                    parameters['sub_ids'][label] = sub_ids_json[label]
        except:
            print(f"Failed to retrieve and save subject id lists from the provided file {sub_ids_json}.")

    # Setup paths for fake test data
    fake_paths = {
        "real_ids": ["fake_data", "real_ids"],
        "range_ids": ["fake_data", "range_ids"],
        "filter_real": ["fake_data", "filter", "filter_real"],
        "filter_range": ["fake_data", "filter", "filter_range"]
    }

    for i in fake_paths:
        fake_paths[i] = os.path.join(paths['data_inputs'], *fake_paths[i])

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
    # Create directories if requested
    if create_dir:
        main_sections.pop("Parameters", None)
        try:
            print(f"\nCreating directories...")
            for main in main_sections:
                for sub in main_sections[main]:
                    if sub not in ['nifti_path', settings_file]:
                        create_directory(main_sections[main][sub])
            print(f"...succesfully created directories.\n")
        except:
            print(f"...failed to create directories.\n")


def get_setup_arguments():
    # fancy command line arguments
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-s", "--sub_ids_json", default=None,
        help="""Optional. Specify the name of a .json file containing a dict
            where keys are group/condition labels and the subject id lists are
            values. A key named 'all' containing all subject ids in the dataset
            is required."""
    )
    parser.add_argument(
        "-n", "--nifti_path", default="",
        help="""The location of subjects' input Nifti data to use for analysis.
        This location will not be defined by default."""
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

    return parser.parse_args()

def main():
    # Get arguments from command line
    args = get_setup_arguments()

    nifti_path = args.nifti_path
    project_path = args.project_path
    create_dir = args.create_dir

    if project_path is None:
        project_path = os.getcwd()
    if args.data_dest is None:
        data_dest = project_path
    else:
        data_dest = args.data_dest

    # TODO: handle os.path incompatible user paths
    # # add starting dash if missing
    # if nifti_path[0] != '/':
    #     nifti_path = '/' + nifti_path

    print(f'Making json settings file...')
    print(f"Nifti path: \n--> '{nifti_path}'")
    print(f"Project path: \n--> '{project_path}'")
    print(f"Data destination: \n--> '{data_dest}'")
    print("\nPlease confirm that these are the correct directories for your analyses.")

    # Make the settings file!
    create_script_settings(nifti_path, project_path, data_dest, create_dir,
                            args.sub_ids_json)

if __name__ == "__main__":
    main()
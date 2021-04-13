#!/usr/bin/env python3

import os
import sys
import argparse
import configparser
import json

from local_intersubject_pkg.tools import create_directory

def create_script_settings(file_type, nifti_path, project_path, data_dest, create_dir):
    """
    Create a configuration file containing filepaths, parameters, and other
    persistent info that are used by this analysis. The resulting file is
    a JSON file that can also be changed manually.

    Its filepaths represent the default directory structure of this project. 

    Currently both .json and .ini are allowed, but .json is preferred. 
    """
    # ========================================================
    # Define paths

    # script settings file name
    settings_file = "script_settings" + file_type
    
    # Create dicts of filepaths
    paths = {
        "settings_path": settings_file,
        "project_path": project_path,

        "data_dest": data_dest,
        "data_inputs": os.path.join(data_dest, "data", "input"),
        "data_outputs": os.path.join(data_dest, "data", "output"),

        "nifti_path": nifti_path
    }
    paths['npy_path'] = os.path.join(paths['data_inputs'], 'npy_data')

    # Setup parameters needed between scripts
    parameters = {
        "datasize" : (4, 4, 4, 10)
    }

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
    # Save script settings file
    main_sections = {
        'Paths': paths, 
        'Parameters': parameters, 
        'Fake Data Paths': fake_paths}
    # main_subsections = [paths, parameters, fake_paths]
    
    try:
        # saves as .json or .ini
        if file_type == '.json':

            # Save file
            with open(settings_file, 'w') as outfile:
                json.dump(main_sections, outfile, indent=4)

        elif file_type == '.ini':
            # create configparser object
            config = configparser.ConfigParser()
            
            for main, sub in zip(main_sections, main_subsections):
                config[main] = sub
        
            # Save file
            with open(settings_file, 'w') as outfile:
                config.write(outfile)

        assert os.path.exists(settings_file)
        print(f"...Successfully wrote '{settings_file}' configuration file")
            
    except:
        print(f"...Could not write '{settings_file}' configuration file") 

    # ==========================
    # Create directories if requested
    if create_dir:
        main_sections.pop("Parameters", None)

        for main in main_sections:
            for sub in main_sections[main]:

                path = main_sections[main][sub]
                create_directory(path)


def get_setup_arguments():
    # fancy command line arguments
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-t", "--file_type", choices=['.json', '.ini'],
        default='.json',
        help="Choose whether to create a .json or .ini settings file."
        )
    parser.add_argument(
        "-n", "--nifti_path",
        help="The location of subjects' input Nifti data to use for analysis."
    )
    parser.add_argument(
        "-p", "--project_path", default = cwd,
        help="""Optional; The location of this project. If not provided, 
        then it is defined as the current working directory of this script."""
    )
    parser.add_argument(
        "-d", "--data_dest",
        help="""Optional; The parent data directory destination to accomadate a different
        location from the project directory containing the code. If not provided, then data_dest
        is made the same as project_path."""
    )
    parser.add_argument(
        "-c", "--create_dir",
        help="""Create directory structure as defined in this settings file. If this flag
        is not provided, then directory creation is supressed.""",
        action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    cwd = os.getcwd()

    # Get arguments from command line
    args = get_setup_arguments()

    file_type = args.file_type
    nifti_path = args.nifti_path
    project_path = args.project_path
    create_dir = args.create_dir

    if args.data_dest == None:
        try:
            assert project_path != None
            data_dest = project_path
        except:
            print(f"{project_path} was not a valid entry")
    else:
        data_dest = args.data_dest

    # TODO: handle os.path incompatible user paths
    # # add starting dash if missing
    # if nifti_path[0] != '/':
    #     nifti_path = '/' + nifti_path

    print(f'Making {file_type} type settings file...')
    print(f"Nifti path: \n--> '{nifti_path}'")
    print(f"Project path: \n--> '{project_path}'")
    print(f"Data destination: \n--> '{project_path}'")

    # Make the settings file!
    create_script_settings(file_type, nifti_path, project_path, data_dest, create_dir)


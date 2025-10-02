"""
MotionPype
© Guillermo Hidalgo-Gadea, Department of Biopsychology
Institute of Cognitive Neuroscience, Ruhr University Bochum

source: https://gitlab.ruhr-uni-bochum.de/ikn/motionpype.git
Licensed under GNU Lesser General Public License v2.1
"""

# list of all required libraries, reduce if possible 
import os
import re
import cv2
import toml
import socket
import shutil
import pathlib
import deeplabcut
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# own module components
import utils


def read_config(config_path):
    '''
    Returns dictionary from toml config file
    '''
    config = toml.load(config_path)
    return config

def change_toml(edits, config_path, output=False):
    '''
    Edits the toml config file in config_path with edits dictionary
    '''
    configfile = toml.load(config_path)
    # edit configfile from dict
    for key, value in edits.items():
        configfile[key] = value
    # dump new configfile
    with open(config_path, 'w') as f:
        new_toml_string = toml.dump(configfile, f)
    if output:
        print(new_toml_string)

# check video extension by content
def check_videotype(config_path, subdir):
    '''
    Returns the file extension of video files in specific subdirectories
    TODO this returns h5 files once analysis started
    '''
    projectpath = os.path.dirname(config_path)
    files = utils.scrapdirbystring(projectpath, subdir, output=False)
    extensions = [pathlib.Path(file).suffix for file in files]
    videotype = set(extensions).pop().replace('.', '')
    if len(set(extensions)) > 1:
        print(f'Found multiple video extensions: {videotype}')
    return videotype

# create project structure
def create_projectstructure(project_name, project_directory, session_list, reference_nesting=0, calibration_nesting=0, output = False):
    '''
    Creates a new directory tree following anipose structure, see here: https://anipose.readthedocs.io/en/latest/start3d.html
    Note: For nesting > 1 please provide the 'session/trial' format in the session list instead of 'session_trial'
    '''
    # create session dirs from list
    for session in session_list:
        # make videos-raw folder for behavior
        videopath = os.path.join(project_directory, project_name, session, 'videos-raw')
        if not os.path.exists(videopath):
            try:
                os.makedirs(videopath)
            except:
                print(f'Error creating directory: {videopath}')
                print(f'Please check project structure and permissions in: {project_directory}')

        # structure videos-ref folder 
        if reference_nesting == 0:
            # same level
            refpath = os.path.join(project_directory, project_name, session, 'videos-ref')
        elif reference_nesting == 1:
            # one level higher
            refpath = os.path.join(project_directory, project_name, os.path.dirname(session), 'videos-ref')
        elif reference_nesting == 2:
            # two levels higher
            refpath = os.path.join(project_directory, project_name, os.path.dirname(os.path.dirname(session)), 'videos-ref')
        elif reference_nesting == 2:
            # three levels higher
            refpath = os.path.join(project_directory, project_name, os.path.dirname(os.path.dirname(os.path.dirname(session))), 'videos-ref')
        else:
            refpath = os.path.join(project_directory, project_name)
            print(f'No videos-ref, if missing check reference_nesting')
        
        # make videos-ref folder for reference
        if not os.path.exists(refpath):
            try:
                os.makedirs(refpath)
            except:
                print(f'Error creating directory: {refpath}')
                print(f'Please check project structure and permissions in: {project_directory}')

        # structure calibration folder 
        if calibration_nesting == 0:
            # same level
            calibrationpath = os.path.join(project_directory, project_name, session, 'videos-cal')
        elif calibration_nesting == 1:
            # one level higher
            calibrationpath = os.path.join(project_directory, project_name, os.path.dirname(session), 'videos-cal')
        elif calibration_nesting == 2:
            # two levels higher
            calibrationpath = os.path.join(project_directory, project_name, os.path.dirname(os.path.dirname(session)), 'videos-cal')
        elif calibration_nesting == 2:
            # three levels higher
            calibrationpath = os.path.join(project_directory, project_name, os.path.dirname(os.path.dirname(os.path.dirname(session))), 'videos-cal')
        else:
            calibrationpath = os.path.join(project_directory, project_name)
            print(f'No calibration, if missing check calibration_nesting')
        
        # make calibration folder
        if not os.path.exists(calibrationpath):
            try:
                os.makedirs(calibrationpath)
            except:
                print(f'Error creating directory: {calibrationpath}')
                print(f'Please check project structure and permissions in: {project_directory}')

    # save project directory and update working directory
    projectpath = os.path.join(project_directory, project_name)
    os.chdir(projectpath)

    if output:
        print(f'Anipose Project created or updated: {os.getcwd()}')

    return projectpath

def populate_project(projectpath, behavior_path, calibration_path, reference_path = '', separators = r'_|\\|/',copy = True):
    '''
    Populates an existing project structure with videos from paths provided
    Note: Use copy = False to move files, moving can be much faster on same disk
    Note: video paths can only contain videos and no further files or folders
    Note: matching videos breaks if sessions have additional info not found in video
    # this works for now, but may need fixing to generalize to other project structures
    '''
    # select videos, if paths provided
    try:
        behavior_videos = os.listdir(behavior_path)
    except:
        behavior_videos = []
    try:
        calibration_videos = os.listdir(calibration_path)
    except:
        calibration_videos = []
    try:
        reference_videos = os.listdir(reference_path)
    except:
        reference_videos = []

    # find session directories in projectpath
    dirs = utils.scrapfoldersindir(projectpath, output=False)
    videos_raw = [i for i in dirs if 'videos-raw' in i]
    videos_cal = [i for i in dirs if 'videos-cal' in i]
    videos_ref = [i for i in dirs if 'videos-ref' in i]

    # BEHAVIOR
    for dir in videos_raw:
        session = os.path.dirname(os.path.relpath(dir, projectpath))
        key1 = key2 = key3 = ''
        try: 
            key1 = re.split(separators, session)[-1]
            key2 = re.split(separators, session)[-2]
            key3 = re.split(separators, session)[-3]
        except:
            pass
        
        # find matching videos
        if key3:
            src_behavior_videos = [video for video in behavior_videos if key1 in video if key2 in video if key3 in video]
        elif key2:
            src_behavior_videos = [video for video in behavior_videos if key1 in video if key2 in video]
        elif key1:
            src_behavior_videos = [video for video in behavior_videos if key1 in video]
        else:
            src_behavior_videos = []

        # copy or move videos
        for video in src_behavior_videos:
            src_file = os.path.join(behavior_path, video)
            dst_file = os.path.join(dir, video)
            if copy:
                print(f'Copying files to {dst_file}')
                shutil.copyfile(src_file, dst_file)
            else:
                print(f'Moving files to {dst_file}')
                os.rename(src_file, dst_file)

    # CALIBRATION
    for dir in videos_cal:
        session = os.path.dirname(os.path.relpath(dir, projectpath))
        key1 = key2 = key3 = ''
        try: 
            key1 = re.split(separators, session)[-1]
            key2 = re.split(separators, session)[-2]
            key3 = re.split(separators, session)[-3]
        except:
            pass
        
        # find matching videos
        if key3:
            src_calibration_videos = [video for video in calibration_videos if key1 in video if key2 in video if key3 in video]
        elif key2:
            src_calibration_videos = [video for video in calibration_videos if key1 in video if key2 in video]
        elif key1:
            src_calibration_videos = [video for video in calibration_videos if key1 in video]
        else:
            src_calibration_videos = []

        # copy or move videos
        for video in src_calibration_videos:
            src_file = os.path.join(calibration_path, video)
            dst_file = os.path.join(dir, video)
            if copy:
                print(f'Copying files to {dst_file}')
                shutil.copyfile(src_file, dst_file)
            else:
                print(f'Moving files to {dst_file}')
                os.rename(src_file, dst_file)
    
    # REFERENCE
    for dir in videos_ref:
        session = os.path.dirname(os.path.relpath(dir, projectpath))
        key1 = key2 = key3 = ''
        try: 
            key1 = re.split(separators, session)[-1]
            key2 = re.split(separators, session)[-2]
            key3 = re.split(separators, session)[-3]
        except:
            pass
        
        # find matching videos
        if key3:
            src_reference_videos = [video for video in reference_videos if key1 in video if key2 in video if key3 in video]
        elif key2:
            src_reference_videos = [video for video in reference_videos if key1 in video if key2 in video]
        elif key1:
            src_reference_videos = [video for video in reference_videos if key1 in video]
        else:
            src_reference_videos = []

        # copy or move videos
        for video in src_reference_videos:
            src_file = os.path.join(reference_path, video)
            dst_file = os.path.join(dir, video)
            if copy:
                print(f'Copying files to {dst_file}')
                shutil.copyfile(src_file, dst_file)
            else:
                print(f'Moving files to {dst_file}')
                os.rename(src_file, dst_file)

    return


def create_config(projectpath, output = False):
    '''
    Creates a config.toml file with pre-defined structure
    see original from https://github.com/lambdaloop/anipose/blob/master/anipose/anipose.py
    TODO why path and project? remove path?
    '''

    config = {
        'project': projectpath,
        #'path': projectpath, # TODO:  this is useed by anipose, but needs to be updated for remote server...?
        'model_folder': "G:\\ForagingPlatformsArena_local\\Triangulation\\DLC_PigeonSuperModel_effnet_b0-PigeonSuperModel.com-2023-04-21",
        'nesting': 1, 
        'video_extension': 'avi', 
        'converted_video_speed': 1,
        'local_anaconda': "C:/Users/hidalggc/Anaconda3/Scripts/activate.bat C:/Users/hidalggc/Anaconda3",
        'remote_anaconda': "C://ProgramData/Anaconda3/Scripts/activate.bat C://ProgramData/Anaconda3",
        'motionpype': {
            'beh_model_folder': "G:\\ForagingPlatformsArena_local\\Triangulation\\DLC_PigeonSuperModel_effnet_b0-PigeonSuperModel.com-2023-04-21",
            'ref_model_folder': "G:\\ForagingPlatformsArena_local\\Triangulation\\DLC_RefModel_HexArena_resnet_50-PigeonSuperModel.com-2023-04-21",
        },
        'clone': {
            'cloned_source': "G:\\ForagingPlatformsArena_local\\Triangulation",
            'cloned_destination': "\\\\compute.ikn.psy.rub.de\\D$\\UserData\\Guillermo\\ForagingPlatforms_Triangulation_server",
            'remote_path': "D:\\UserData\\Guillermo\\ForagingPlatforms_Triangulation_server",
        },
        'process': {
            'last_host': "compute-ikn.serverhosting.ruhr-uni-bochum.de",
            'last_ip': "134.147.231.123",
            'last_process': "filter",
        },
        'pipeline': {
            'videos_raw': 'videos-raw',
            'calibration_videos': 'videos-cal',
            'calibration_results': 'videos-cal',
            'pose_2d': 'pose-2d',
            'pose_2d_filter': 'pose-2d-filtered',
            'pose_3d': 'pose-3d',
            'pose_3d_filter': 'pose-3d-filtered',
        },
        'calibration': {
            'animal_calibration': False,
            'calibration_init': None,
            'fisheye': False,
        },
        'manual_verification': {
            'manually_verify': False,
        },
        'triangulation': {
            'triangulate': True, 
            'cam_regex': '_cam([A-Z])$', 
        },
        'filter': {
            'enabled': True,
        },
        'filter3d': {
            'enabled': True,
        }
    }

    # make config.toml file
    config_path = os.path.join(projectpath, 'config.toml')
    with open(config_path, 'w') as f:
        new_toml_string = toml.dump(config, f)
    
    if output:
        print(new_toml_string)

    return config_path

def clone_anipose_project(config_path, client, dest_path, videos = False):
    '''
    Clones an existing project using robocopy from a bat file.
    Note: requires specific format for client and destination path as:
    `\\compute.ikn.psy.rub.de`
    `\\D$\\UserData\\Guillermo\\Triangulation_server` 
    '''

    # mapping remote and local paths
    destination = os.path.join(client, dest_path)
    if '$' in destination:
        dirs = os.path.realpath(destination).replace('$', ':').split(os.sep)
    else:
        dirs = os.path.realpath(destination).split(os.sep)
    disk = [element for element in dirs if ':' in element].pop()
    remote_path = os.path.realpath("\\".join(dirs[dirs.index(disk):]))

    # path to the config on the remote
    remote_config = os.path.join(remote_path, 'config.toml')
    
    # absolute path between machines
    cloned_config = os.path.join(destination, 'config.toml')

    # add local and remote paths to config
    source = toml.load(config_path)['project']
    edits = {
        'clone':{
            'remote_path': remote_path,
            'cloned_source': source,
            'cloned_destination': destination,
            },
        }
    change_toml(edits, config_path)

    # add remote anaconda 
    try:
        remote_anaconda = toml.load(config_path)['remote_anaconda']
    except:
        remote_anaconda = ''
    if remote_anaconda:
        pass
    else:
        anacondacommand = find_anaconda(config_path, directories = [os.path.join(client, r'C$\\Users'),os.path.join(client, r'C%\\ProgramData'),], query = 'Anaconda3', output = False)

        edits = {'remote_anaconda': anacondacommand}
        change_toml(edits, config_path)

    if videos:
        # create executable bat file 
        batfile = os.path.join(source, 'motionpype_processes', "clone_project.bat")
        if not os.path.exists(os.path.dirname(batfile)):
            os.makedirs(os.path.dirname(batfile))
        myBat = open(batfile,'w+')
        myBat.write(f'''
@ echo off
:: set source and destination variables
set SOURCE={source}
set DESTINATION={destination}
:: start robocopy clone
robocopy %SOURCE% %DESTINATION% /E /MT:100 /TEE /ETA /V /A-:SH /unilog+:%SOURCE%\motionpype_processes\RobocopyTransfer.log
:: start robocopy mirror back
robocopy %DESTINATION% %SOURCE% /E /MT:100 /TEE /ETA /V /A-:SH /unilog+:%SOURCE%\motionpype_processes\RobocopyTransfer.log
call pause
''')
        myBat.close()
    else:
        # create executable bat file without videos
        batfile = os.path.join(source, 'motionpype_processes', "clone_project.bat")
        if not os.path.exists(os.path.dirname(batfile)):
            os.makedirs(os.path.dirname(batfile))
        myBat = open(batfile,'w+')
        myBat.write(f'''
@ echo off
:: set source and destination variables
set SOURCE={source}
set DESTINATION={destination}
:: start robocopy clone
robocopy %SOURCE% %DESTINATION% /E /MT:100 /TEE /ETA /V /A-:SH /XF "*.mp4" /XF "*.avi" /unilog+:%SOURCE%\motionpype_processes\RobocopyTransfer.log
:: start robocopy mirror back
robocopy %DESTINATION% %SOURCE% /E /MT:100 /TEE /ETA /V /A-:SH /XF "*.mp4" /XF "*.avi" /unilog+:%SOURCE%\motionpype_processes\RobocopyTransfer.log
call pause
''')
        myBat.close()
    
    # execute clone_project
    os.startfile(batfile)
    
    return print(f'Cloning project: {source} \n to remote path: {remote_path} on {client}')

def pull_cloned_project(config_path, videos = False):
    '''
    Populates the cloned source project with any updates from the remote clone
    Note: clone_anipose_project() already mirrors back after each push...
    '''
    projectpath = toml.load(config_path)['project']
    destination = toml.load(config_path)['clone']['cloned_source']
    source = toml.load(config_path)['clone']['cloned_destination']

    if videos:
        # create executable bat file 
        batfile = os.path.join(projectpath, 'motionpype_processes', "pull_clone_project.bat")
        if not os.path.exists(os.path.dirname(batfile)):
            os.makedirs(os.path.dirname(batfile))
        myBat = open(batfile,'w+')
        myBat.write(f'''
@ echo off
:: set source and destination variables
set SOURCE={source}
set DESTINATION={destination}
:: start robocopy pull
robocopy %SOURCE% %DESTINATION% /E /MT:100 /TEE /ETA /V /A-:SH /XF "config.toml" /unilog+:%DESTINATION%\motionpype_processes\RobocopyTransfer.log
call pause
''')
        myBat.close()
    else:
        # create executable bat file without videos
        batfile = os.path.join(projectpath, 'motionpype_processes', "pull_clone_project.bat")
        if not os.path.exists(os.path.dirname(batfile)):
            os.makedirs(os.path.dirname(batfile))
        myBat = open(batfile,'w+')
        myBat.write(f'''
@ echo off
:: set source and destination variables
set SOURCE={source}
set DESTINATION={destination}
:: start robocopy pull
robocopy %SOURCE% %DESTINATION% /E /MT:100 /TEE /ETA /V /A-:SH /XF "config.toml" /XF "*.mp4" /XF "*.avi" /unilog+:%DESTINATION%\motionpype_processes\RobocopyTransfer.log
call pause
''')
        myBat.close()
    
    # execute clone_project
    os.startfile(batfile)
    
    return print(f'Pulling project from {source} \n to: {destination}')


def find_anaconda(config_path, directories = ['C:\\Users','C:\\ProgramData',], query = 'Anaconda3', overwrite = False, output = False):
    '''
    Returns the anaconda activation command and overwrites config parameters
    '''

    # check existing anaconda paths in config
    try:
        local_anaconda = toml.load(config_path)['local_anaconda']
    except:
        local_anaconda = ''
    try:
        remote_anaconda = toml.load(config_path)['remote_anaconda']
    except:
        remote_anaconda = ''

    # find anaconda paths from directories
    for dir in directories:
        try:
            anacondafiles.append(utils.scrapdirbystring(dir, query, output = output))
        except:
            anacondafiles = utils.scrapdirbystring(dir, query, output = output)

    anacondapath = os.path.dirname(anacondafiles[0])
    if '$' in anacondapath:
        parts = anacondapath.split('$')
        anacondapath = parts[0][-1] + ':' + parts[-1]
        remote = True
    else:
        remote = False

    activation = os.path.join(anacondapath, 'Scripts', 'activate.bat')
    anacondacommand = (activation + ' ' + anacondapath).replace(os.sep, '/')

    if output:
        print(f'Found Anaconda path in {anacondapath}')

    # update config instead of returning file
    if remote:
        edits = {'remote_anaconda': anacondacommand}
        if output:
            print(f'Found remote_anaconda: {anacondacommand}')
    else:
        edits = {'remote_anaconda': anacondacommand}
        if output:
            print(f'Found local_anaconda: {anacondacommand}')

    # check if already exists before updating
    if local_anaconda:
        if not overwrite:
            pass
        else:
            change_toml(edits, config_path)
    else:
        change_toml(edits, config_path)
    
    if remote_anaconda:
        if not overwrite:
            pass
        else:
            change_toml(edits, config_path)
    else:
        change_toml(edits, config_path)
    
    return anacondacommand

def execute_anipose(settings, command, config_path, client = '', output = False):
    '''
    Executes anipose commands via bat files, if remote the bat file is triggered via psexec.
    Note: psexec needs to be installed and should not be blocked by antivirus software in client.
    Note: running batch files may require elevated privileges
    '''
    
    # check validity of remote client , e.g., `client = r'\\compute.ikn.psy.rub.de'`
    if client:
        try:
            ipaddr = socket.gethostbyname(os.path.basename(client))
            host = socket.gethostbyaddr(ipaddr)[0]
            remote = True
        except:
            # if unreachable, default to local machine
            host = socket.gethostname()
            ipaddr = socket.gethostbyname(host)
            remote = False
    else:
        # if not provided, default to local machine
        host = socket.gethostname()
        ipaddr = socket.gethostbyname(host)
        remote = False

    # access project parameters and configure paths
    if remote:
        cloned_config_path = os.path.join(toml.load(config_path)['clone']['cloned_destination'], 'config.toml')
        change_toml(settings, cloned_config_path)
        path = toml.load(config_path)['clone']['remote_path']
        disk = os.path.splitdrive(path)[0]
        change_toml({'project': path}, cloned_config_path)
        try:
            anaconda = toml.load(config_path)['remote_anaconda']
        except:
            print(f'`remote_anaconda` parameter missing in config.toml...')
            anaconda = find_anaconda(config_path, directories = [os.path.join(client, r'C$\\Users'),os.path.join(client, r'C%\\ProgramData'),], query = 'Anaconda3', output = False)

    else:
        change_toml(settings, config_path)
        path = toml.load(config_path)['project']
        disk = os.path.splitdrive(path)[0]
        try:
            anaconda = toml.load(config_path)['local_anaconda']
        except:
            print(f'`local_anaconda` parameter missing in config.toml...')
            anaconda = find_anaconda(config_path, directories = ['C:\\Users','C:\\ProgramData',], query = 'Anaconda3', output = False)

    # update config settings
    edits = {'process': {
                'last_host': host, 
                'last_ip': ipaddr, 
                'last_process': command
                }
            }
    
    change_toml(edits, config_path)

    # create batch file with commands
    projectpath = toml.load(config_path)['project']

    batfile = os.path.join(projectpath, 'motionpype_processes', f"execute_anipose_{command}.bat")
    if not os.path.exists(os.path.dirname(batfile)):
        os.makedirs(os.path.dirname(batfile))
    myBat = open(batfile,'w+')
    myBat.write(f'''
@echo off
:: start anaconda terminal 
call {anaconda}
:: activate environment
call activate anipose
:: move to project path
{disk}
cd {path}
:: run command
call anipose {command}
pause
''')
    myBat.close()

    # create remote psexec bat
    psexecBAT =  os.path.join(projectpath, 'motionpype_processes', "psexec.bat")
    if not os.path.exists(os.path.dirname(psexecBAT)):
        os.makedirs(os.path.dirname(psexecBAT))
    myBat = open(psexecBAT,'w+')
    myBat.write(f'''
psexec {client} -c -f {batfile}
pause
''')
    myBat.close()

    # execute via commands via bat or psexec
    if remote:
        os.startfile(psexecBAT)
    else:
        os.startfile(batfile)
    
    return print(f'Running anipose {command} on {host}...')

def project_overview(config_path):
    '''
    Returns descriptives of ongoing project progress
    TODO
    '''
    # check config parameters
    config = toml.load(config_path)
    project_dir = os.path.dirname(config_path)
    project_path = config['project']
    beh_model_folder = config['motionpype']['beh_model_folder']
    ref_model_folder = config['motionpype']['ref_model_folder']

    try:
        # optional if executed
        last_process = config['process']['last_process']
        last_host = config['process']['last_host']
        last_ip = config['process']['last_ip']
        remote_anaconda = config['remote_anaconda']
        local_anaconda = config['local_anaconda']

        # optional if cloned
        remote_path = config['clone']['remote_path']
        cloned_source = config['clone']['cloned_source']
        cloned_destination = config['clone']['cloned_destination']
    except:
        pass    
    # check paths
    if project_dir == project_path:
        print(f'Project active in {project_path}')
        path = project_dir
    else:
        print(f'Project path does NOT correspond to location of config.toml!')
        ask = input(f'Update {project_path} with {project_dir} in config.toml? [y/n]') # input in vs code is up...
        if 'y' in ask:
            # update config.toml
            edits = {'project': project_dir}
            change_toml(edits, config_path)
            path = project_dir
        else:
            print('Config.toml NOT updated.')
            print(f'Project active in {project_path}')
            path = project_dir

    # report project size
    sessions = [session for session in utils.scrapfoldersindir(path, output = False) if config['pipeline']['calibration_videos'] in session]
    trials = [session for session in utils.scrapfoldersindir(path, output = False) if config['pipeline']['videos_raw'] in session]
    print(f'Found {len(sessions)} sessions and {len(trials)} trials')
    
    calibrations = []
    numcam = []
    for session in sessions:
        numcam.append(len([video for video in os.listdir(session) if not 'toml' in video if not 'pickle' in video]))
        calibrations.append(len([file for file in os.listdir(session) if 'toml' in file]))
    print(f'Found {set(numcam)} videos per calibration, with {sum(calibrations)} out of {len(calibrations)} sessions already calibrated')
    calibrationfiles = [os.path.join(session, 'calibration.toml') for session in sessions]
    calerror = []
    for file in calibrationfiles:
        try: 
            calerror.append(toml.load(file)['metadata']['error'])
        except:
            pass
    bootstrap = []
    for i in range(1000):
        bootstrap_sample = pd.DataFrame(calerror).sample(len(calerror), replace= True)
        bootstrap.append(bootstrap_sample.mean().values)
    print(f'- average calibration error of {np.mean(calerror):.2f}')
    print(f'- bootstraped 95%-CI [{np.percentile(bootstrap, 2.5):.2f} - {np.percentile(bootstrap, 97.5):.2f}]')

    numvideos = []
    videos_raw = [os.listdir(trial) for trial in trials]
    for videos in videos_raw:
        numvideos.append(len(videos))
    videoslist = [element for sublist in videos_raw for element in sublist]
    print(f'Found {len(videoslist)} videos, ({int(100*len(videoslist)/(len(trials)*max(set(numcam))))} % of expected with {max(set(numcam))} cameras and {len(trials)} trials)')
    print(f'Found {set(numvideos)} number of behavior videos per trial')
    posefiles = [file for file in utils.scrapdirbystring(path, config['pipeline']['pose_2d'], output=False) if '.h5' in file if 'filtered' not in file if 'merged' not in file]
    print(f'Found {len(posefiles)} analyzed videos ({int(100*len(posefiles)/len(videoslist))} % of all {len(videoslist)} videos)')
    triangulatedfiles = [file for file in utils.scrapdirbystring(path, config['pipeline']['pose_3d'], output=False) if '.csv' in file]
    print(f'Found {len(triangulatedfiles)} triangulated trials ({int(100*len(triangulatedfiles)/len(trials))} % of all {len(trials)} trials)')

    return

def merge_referenceframes(config_path, overwrite = False, output = True):
    '''
    Returns h5 files with merged coordinates of behavior and reference frame for each pose file
    '''
    # read config parameters
    try:
        config = read_config(config_path)
        projectpath = config['project']
        pose_dir = config['merge']['behavior']
        ref_dir = config['merge']['reference']
        merged_dir = config['merge']['output']
        nesting = config['merge']['nesting']
        cam_regex = config['triangulation']['cam_regex']
    except:
        print('Error merging reference frames, check ["merge"] in config file')

    # scrap sessions
    poses = [pose for pose in utils.scrapdirbystring(projectpath, pose_dir, output = output) if '.h5' in pose]
    references = [ref for ref in utils.scrapdirbystring(projectpath, ref_dir, output = False) if '.h5' in ref]
    sessions = list(set([os.path.abspath(pose.split(pose_dir)[0]) for pose in poses])) # get path above pose_dir

    # match files per session
    for session in sessions:
        try:
            # get nesting
            level = session
            for i in range(nesting):
                level = os.path.dirname(session) # recursive path shortening

            # subset files
            pose_files = [pose for pose in poses if session in os.path.abspath(pose)]
            ref_files = [ref for ref in references if level in os.path.abspath(ref)]
            
            # create merged_dir for output
            outputdir = os.path.join(session, merged_dir)
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            else:
                pass
            
            # match files 
            for ref_file in ref_files:
                try:
                    if output:
                        print(f'\nMerging reference: {ref_file}')

                    # match data by camera
                    match = re.search(cam_regex, ref_file.split('.')[0]).group(0)
                    beh_file = [file for file in pose_files if match in file][0]
                    outputfile = os.path.join(outputdir, os.path.basename(beh_file))
                    if output:
                        print(f'With pose data: {beh_file}')

                    # check if output file already existing
                    if os.path.exists(outputfile):
                        if overwrite:
                            ignore = False
                        else:
                            ignore =True
                    else:
                        ignore = False

                    if not ignore:
                        # read data
                        reference = pd.read_hdf(ref_file)
                        behavior = pd.read_hdf(beh_file)
                        ref_scorer = reference.columns.levels[0][0]
                        beh_scorer = behavior.columns.levels[0][0]
                        
                        # calculate stable reference points
                        if output:
                            print('... calculating stable reference points')

                        median = pd.DataFrame(reference.median(axis = 0, skipna=True)).T
                        stable = pd.concat([median]*len(behavior), ignore_index=True).rename(columns={ref_scorer:beh_scorer})
                        
                        # merge reference and behavior data
                        merged = pd.concat([stable, behavior], axis=1)

                        # save data
                        merged.to_hdf(outputfile, key = "df_with_missing", mode='w')
                        if output:
                            print(f'Done! Merged Data: {os.path.basename(outputfile)}')
                    else:
                        print(f'Merged Data already exists, skipping.')
    
                except:
                    print(f'Error merging files for {session}\n')

        except:
            print(f'No file match for {session}\n')

    return

def spatial_median_tracking():
    '''
    Returns an hdf dataframe in DLC format with spatial-median filtered tracking from DLC files with `num_outputs` > 1
    This is a faster alternative to the viterbi filter in anipose
    TODO
    '''
    # loop over project and calculate spatial median from unfiltered data
    projectpath = r'H:\StopSignalSkinnerbox_local\Skinnerbox_Triangulation_SCM'
    pose_dir = 'pose-2d'
    output_dir = 'spatial-median'

    for session in os.listdir(projectpath):
        try:
            print(f'Processing session: {session}')
            # get all pose files
            pose_files = os.listdir(os.path.join(projectpath, session, pose_dir))
            unfiltered_files = [file for file in pose_files if '.h5' in file]
            outputdir = os.path.join(projectpath, session, output_dir)

            # create output dir
            if os.path.isdir(outputdir):
                print('Spatial median filter already exists.')
                pass
            else:
                os.makedirs(outputdir)

                # loop over files
                for file in unfiltered_files:
                    print(f'Processing file: {file}')
                    filepath = os.path.join(projectpath, session, pose_dir, file)
                    df = pd.read_hdf(filepath)
                    
                    # get multiindex
                    scorer = df.columns.get_level_values(0)[0]
                    unique_bodyparts=[]
                    for part in df.columns.get_level_values(1):
                        if part not in unique_bodyparts:
                            unique_bodyparts.append(part)

                    med_file = []
                    # split by bodypart
                    for bodypart in unique_bodyparts:
                        bodypartdata = df[scorer, bodypart]

                        med_bodypart = []
                        # split by coord
                        for coordinate in ['x', 'y', 'likelihood']:
                            coord_select = coord_select = [coord for coord in bodypartdata.columns if coordinate in coord]
                            # calculate median
                            med_bodypart.append(np.nanmedian(bodypartdata[coord_select], axis=1))
                        
                        # reconstruct median data frame per bodypart
                        df_med = pd.DataFrame(np.array(med_bodypart).T, columns=pd.MultiIndex.from_product([[scorer], [bodypart], ['x','y','likelihood']]))
                        med_file.append(df_med)
                        
                    # reconstruct median data frame per file
                    MedianFiltered = pd.concat(med_file, axis=1, ignore_index=False)
                    MedianFiltered.columns.set_names(['scorer', 'bodyparts', 'coords'], inplace=True)

                    # save medians
                    outputfilename = os.path.join(outputdir, 'median_' + file)
                    print(f'Saving output file: {outputfilename}')
                    MedianFiltered.to_hdf(outputfilename, key ='df', mode='w')
                    print('')

        except:
            print(f'No files found in {pose_dir}')

    return

def expand_project(config_path, newsessions, calibration_nesting = False, reference_nesting = False, output = True):
    '''
    Expands existing project by new sessions
    '''
    # read paths and nesting structure from existing project
    projectpath = toml.load(config_path)['project']
    if not calibration_nesting:
        sessions = os.listdir(projectpath)
        reference_nesting = []
        calibration_nesting = []
        for session in sessions:
            raw_dir = [os.path.dirname(file) for file in utils.scrapdirbystring(os.path.join(projectpath, session), "videos-raw", output=False)]
            cal_dir = [os.path.dirname(file) for file in utils.scrapdirbystring(os.path.join(projectpath, session), "videos-cal", output=False)]
            ref_dir = [os.path.dirname(file) for file in utils.scrapdirbystring(os.path.join(projectpath, session), "videos-ref", output=False)]
            try:
                reference_nesting.append(raw_dir[0].count(os.path.sep) - ref_dir[0].count(os.path.sep))
                calibration_nesting.append(raw_dir[0].count(os.path.sep) - cal_dir[0].count(os.path.sep))
            except:
                pass
            
        reference_nesting = int(list(set(reference_nesting)).pop())
        calibration_nesting = int(list(set(calibration_nesting)).pop())
        # TODO what if sessions have inconsistent nesting structures?
    else:
        pass

    # create session dirs from new list
    for session in newsessions:
        # make videos-raw folder for behavior
        videopath = os.path.join(projectpath, session, 'videos-raw')
        if not os.path.exists(videopath):
            try:
                os.makedirs(videopath)
            except:
                print(f'Error creating directory: {videopath}')
                print(f'Please check project structure and permissions in: {projectpath}')

        # structure videos-ref folder 
        if reference_nesting == 0:
            # same level
            refpath = os.path.join(projectpath, session, 'videos-ref')
        elif reference_nesting == 1:
            # one level higher
            refpath = os.path.join(projectpath, os.path.dirname(session), 'videos-ref')
        elif reference_nesting == 2:
            # two levels higher
            refpath = os.path.join(projectpath, os.path.dirname(os.path.dirname(session)), 'videos-ref')
        elif reference_nesting == 2:
            # three levels higher
            refpath = os.path.join(projectpath, os.path.dirname(os.path.dirname(os.path.dirname(session))), 'videos-ref')
        else:
            refpath = os.path.join(projectpath)
            print(f'No videos-ref, if missing check reference_nesting')
        
        # make videos-ref folder for reference
        if not os.path.exists(refpath):
            try:
                os.makedirs(refpath)
            except:
                print(f'Error creating directory: {refpath}')
                print(f'Please check project structure and permissions in: {projectpath}')

        # structure calibration folder 
        if calibration_nesting == 0:
            # same level
            calibrationpath = os.path.join(projectpath, session, 'videos-cal')
        elif calibration_nesting == 1:
            # one level higher
            calibrationpath = os.path.join(projectpath, os.path.dirname(session), 'videos-cal')
        elif calibration_nesting == 2:
            # two levels higher
            calibrationpath = os.path.join(projectpath, os.path.dirname(os.path.dirname(session)), 'videos-cal')
        elif calibration_nesting == 2:
            # three levels higher
            calibrationpath = os.path.join(projectpath, os.path.dirname(os.path.dirname(os.path.dirname(session))), 'videos-cal')
        else:
            calibrationpath = os.path.join(projectpath)
            print(f'No calibration, if missing check calibration_nesting')
        
        # make calibration folder
        if not os.path.exists(calibrationpath):
            try:
                os.makedirs(calibrationpath)
            except:
                print(f'Error creating directory: {calibrationpath}')
                print(f'Please check project structure and permissions in: {projectpath}')

    # save project directory and update working directory
    if output:
        print(f'Anipose Project successfully expanded: {os.getcwd()}')

    return

def calibration_patch():
    '''
    Creates synthetic camera calibration data from similar sessions
    TODO
    '''
    return

def prioritize_sessions():
    '''
    Masks and unmasks sessions to be prioritized during the analysis
    This works because anipose loads the files to process within a session as batch at the beginning of each process, 
    unmasking immediately after does not affect the already started process. 
    TODO: mask and unmask by renaming directories
    '''
    # prioritize GPU analysis by renaming videos-raw-masked

    # prioritize triangulation by renaming videos-cal-masked
    return
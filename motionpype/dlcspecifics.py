"""
MotionPype
© Guillermo Hidalgo-Gadea, Department of Biopsychology
Institute of Cognitive Neuroscience, Ruhr University Bochum

source: https://gitlab.ruhr-uni-bochum.de/ikn/motionpype.git
Licensed under GNU Lesser General Public License v2.1
"""

# list of all required libraries, reduce if possible 
import os
import cv2
import scipy
import shutil
import string
import tarfile
import warnings
import deeplabcut
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
from deeplabcut.create_project import modelzoo
from deeplabcut.utils import auxiliaryfunctions


# own module components
import utils

def merge_dataset(config_path, dataset_url):
    '''
    Add a pre-trained dataset to existing DLC project
    TODO generalize from GitLab and PigeonSuperModel
    '''
    # initialite DLC config.yaml
    target_dir = deeplabcut.auxiliaryfunctions.read_config(config_path)['project_path']
    scorer = deeplabcut.auxiliaryfunctions.read_config(config_path)['scorer']
    original_video_sets = deeplabcut.auxiliaryfunctions.read_config(config_path)['video_sets']

    # Extract dataset from .tar file
    print(f'Extracting tar file to: {target_dir} ...')
    filename, _ = urllib.request.urlretrieve(dataset_url)
    with tarfile.open(filename, mode="r:gz") as tar:
        tar.extractall(target_dir)
    # find dataset downloaded in target_dir 
    dataset_path = [os.path.join(target_dir, path) for path in os.listdir(target_dir) if 'pigeonsupermodel' in path][0]

    # scrap pre-labeled dataset
    print(f'Merging Dataset ...')
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if '.csv' in f:
                try:
                    # rename scorer in .csv file
                    csvfile = os.path.join(root, f)
                    df = pd.read_csv(csvfile)
                    df.columns = ['scorer'] + [scorer for i in range(len(df.columns)-1)]
                    # rename .csv file
                    rename = os.path.join(root, f'CollectedData_{scorer}.csv')
                    df.to_csv(rename, mode="w", index=False)
                    # remove old .csv file
                    os.remove(csvfile)
                except:
                    print('error overwriting csv files, please doublecheck')
                    pass

            elif '.h5' in f:
                # rename scorer in .h5
                h5file = os.path.join(root, f)
                df = pd.read_hdf(h5file)
                df.columns = df.columns.set_levels([scorer], level=0)
                # rename .h5 file
                rename = os.path.join(root, f'CollectedData_{scorer}.h5')
                df.to_hdf(rename, key = "df_with_missing", mode="w")
                # remove old .h5 file
                os.remove(h5file)
                # move frame set to labeled-data
                src = root
                dst = os.path.join(target_dir, 'labeled-data', os.path.basename(root))
                os.rename(src, dst)

    # add dummy videopaths to config.yaml files
    print('Adding video paths to "video_sets" in config.yaml ...')
    framesets = os.listdir(os.path.join(target_dir, 'labeled-data'))
    videos = [os.path.join(target_dir, 'videos',frameset+'.avi') for frameset in framesets]

    # create new dict for video_sets
    video_sets = {}
    for video in videos:
        rel_video_path = str(Path.resolve(Path(video)))
        video_sets[rel_video_path] = {"crop": ", ".join(map(str, [0, 1920, 0, 1080]))}
            
    # expand existing video_sets
    if original_video_sets is None:
        new_video_sets = video_sets
    else:
        new_video_sets = {**original_video_sets, **video_sets}

    # add video paths to project
    statement = '(C) PigeonSuperModel.com'
    edit = {'video_sets': new_video_sets, 
            'PigeonSuperModel_data': statement}
    deeplabcut.auxiliaryfunctions.edit_config(config_path, edit);

    shutil.rmtree(dataset_path)

    return

def dlcframesoverview(config_path):
    '''
    Returns a count of frame sets and labeles as print statement
    TODO exclude labeled frames with skeleton
    '''
    # check dataset
    labels = []
    frames = []

    for root, dirs, files in os.walk(os.path.join(os.path.dirname(config_path), 'labeled-data')):
        for file in files:
            if 'png' in file:
                frames.append(os.path.join(root,file))
            elif 'h5' in file:
                labels.append(os.path.join(root,file)) 
            else:
                pass
    frames = [frame for frame in frames if '_labeled' not in frame]

    print(f'The Project contains {len(labels)} labeled frame sets and {len(frames)} frames.')
    return

def checkdlcevaluation(config_path, plot = False):
    '''
    Plot and tabulate evaluation results from dlc model
    TODO for num = 1 axes object is not iterable
    '''

    # read DLC config.yaml
    config = deeplabcut.auxiliaryfunctions.read_config(config_path)
    projectpath = config['project_path']
    iteration = config['iteration']
    modelname = config['Task']
    pcutoff = config['pcutoff']
    videores = list(config['video_sets'].items())[-1][1]['crop'].split(',')[1:4:2] # look at last video
    
    # read evaluation results
    evals = os.path.join(projectpath, 'evaluation-results', f'iteration-{iteration}', 'CombinedEvaluation-results.csv')
    df = pd.read_csv(evals)
    evaluation_results = df.loc[:, ~df.columns.isin(['Unnamed: 0', '%Training dataset', 'p-cutoff used'])]


    if plot:
        # get parameters
        shuffles = list(set(evaluation_results['Shuffle number']))
        num = len(shuffles)
        errors = [col for col in evaluation_results.columns if 'error' in col]
        ymax = int(evaluation_results[errors].values.max() + evaluation_results[errors].values.max()/10)
        xmax = max(evaluation_results['Training iterations:'])
        
        if num < 4:
            rows = 1
            cols = num
            h = 10
            w = 10 * num
        elif num < 7:
            rows = 2
            cols = 3
            h = 20
            w = 10
        elif num < 10:
            rows = 3
            cols = 3
            h = 30
            w = 30 

        fig, axes = plt.subplots(rows,cols, figsize = (w,h))
        for i, ax in enumerate(axes):
            data = evaluation_results[evaluation_results['Shuffle number']==shuffles[i]].loc[:, ~evaluation_results.columns.isin(['Shuffle number'])]
            data.plot.line(ax = ax,
                    x = 'Training iterations:', 
                    title = f'Shuffle: {shuffles[i]}', 
                    ylabel = 'Pixel Error',
                    xlabel = f'Training Iterations in DLC [p-cutoff = {pcutoff}]',
                    xticks = np.arange(0, xmax, int(xmax / 10)),
                    ylim = (0, ymax),
                    rot = 45)
        # plot evaluation results
        plt.rcParams.update({'font.size':20})
        
        plt.ticklabel_format(axis='both', style='plain')
        plt.suptitle(f'Evaluation of {modelname} model')
        plt.show()

    # check evaluation results
    return evaluation_results

def createrescaledvideosamples(videos, scales):
    '''
    Creates new rescaled video samples with scale in format `960:540` or `960:-1`.
    Useful to test DLC performance on different video resolutions.
    '''
    for video in videos:
        for scale in scales:
            output = os.path.join(os.path.dirname(video), 'resacled_'+scale.replace(':', '_')+'_'+os.path.basename(video))
            command = f"ffmpeg -i {video} -vf scale={scale} {output}"
            os.system(command)
    return

def fixdlclabelindex(config_path, output = False):
    '''
    Fix h5 Multiindex structure overwrighting older DLC labels
    Run this before deeplabcut.dropannotationfileentriesduetodeletedimages()
    '''
    # check dataset
    src = os.path.join(os.path.dirname(config_path), 'labeled-data')
    frames = utils.scrapdirbystring(src, 'png', output=False)
    labels = utils.scrapdirbystring(src, 'h5', output=False)

    if output:
        print(f'The Project contains {len(labels)} frame sets and {len(frames)} extracted frames.')

    # update label index structure
    old_index = []
    for hdf in labels:
        df = pd.read_hdf(hdf)
        new_index = []
        for idx in df.index.to_list():
            if len(idx) > 3:
                new_index.append(tuple(idx.split('\\')))
                old_index.append(hdf)
            else:
                pass
        if len(new_index) >1:
            df.index = pd.MultiIndex.from_arrays(np.transpose(new_index))

            # save new h5 format
            df.to_hdf(hdf, key = "df_with_missing", mode="w")
        else:
            pass
    if output:
        print(f'fixed index structure for {len(set(old_index))} files.')
    return

def manuallyaddnewvideopaths(config_path, video_list=[]):
    '''
    Adds dummy video paths to config.yaml from set of labels or list of videos
    Move additional frames first to 'labeled-data' or define video_list
    '''

    target_dir = deeplabcut.auxiliaryfunctions.read_config(config_path)['project_path']
    original_video_sets = deeplabcut.auxiliaryfunctions.read_config(config_path)['video_sets']
    
    # add dummy videopaths to config.yaml files
    print('Adding video paths to "video_sets" in config.yaml ...')
    if len(video_list) >0:
        videos = [video for video in video_list]
        framesets = [os.path.join(target_dir, 'labeled-data', os.path.splitext(os.path.basename(video))[0]) for video in videos]
    else:
        framesets = os.listdir(os.path.join(target_dir, 'labeled-data'))
        videos = [os.path.join(target_dir, 'videos',frameset+'.avi') for frameset in framesets]

    # create new dict for video_sets
    video_sets = {}
    for video in videos:
        rel_video_path = str(Path.resolve(Path(video)))
        video_sets[rel_video_path] = {"crop": ", ".join(map(str, [0, 1920, 0, 1080]))}
                
    # expand existing video_sets
    if original_video_sets is None:
        new_video_sets = video_sets
    else:
        new_video_sets = {**original_video_sets, **video_sets}

    # add video paths to project
    statement = '(C) PigeonSuperModel.com'
    edit = {'video_sets': new_video_sets, 
            'sourcecode': statement}
    deeplabcut.auxiliaryfunctions.edit_config(config_path, edit);

    # create directories in labeled-data
    for frameset in framesets:
        os.makedirs(frameset)

    return

def overridevideosetsfromlabeleddata(config_path):
    targetdir = deeplabcut.auxiliaryfunctions.read_config(config_path)['project_path']
    datadir = os.path.join(targetdir, 'labeled-data')
    framesets = utils.scrapdirbystring(datadir, 'h5', output=False)
    videos = [os.path.join(targetdir,'videos',os.path.dirname(frameset)+'.avi') for frameset in framesets]

    # create new dict for video_sets
    video_sets = {}
    for video in videos:
        rel_video_path = str(Path.resolve(Path(video)))
        video_sets[rel_video_path] = {"crop": ", ".join(map(str, [0, 1920, 0, 1080]))}
    
    # replace existing video_sets
    statement = '(C) PigeonSuperModel.com'
    edit = {'video_sets': video_sets, 
            'sourcecode': statement}
    deeplabcut.auxiliaryfunctions.edit_config(config_path, edit);

    return

def getboundingbox(df):
    '''
    Returns the coords and the scale of bounding box for every frame in a df
    '''
    # separate labels by coordinate
    scorer = list(set(df.columns.get_level_values('scorer'))).pop()
    xs = []
    ys = []
    for bodypart in list(set(df.columns.get_level_values('bodyparts'))):
        xs.append(df[scorer, bodypart, 'x'])
        ys.append(df[scorer, bodypart, 'y'])

    x_coords = pd.concat(xs, axis=1)
    y_coords = pd.concat(ys, axis=1)
    
    # bounding box and size
    bb = []
    bbs = []

    for row in range(len(df)):
        #get bounding box
        xmin = x_coords.iloc[row].min()
        xmax = x_coords.iloc[row].max()
        ymin = y_coords.iloc[row].min()
        ymax = y_coords.iloc[row].max()
        bb.append(np.array(((xmin,xmax),(ymin,ymax))))

        # calculate scale in px
        bbs.append((xmax-xmin)*(ymax-ymin))

    # add bounding boxes to df
    df_bb = df.copy()
    df_bb[scorer, 'boundingbox', 'coords'] = bb
    df_bb[scorer, 'boundingbox', 'scale'] = bbs

    return df_bb

def calculatedominantcolors(img):
    '''
    Returns dominant color patches from image
    '''
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    indices = np.argsort(counts)[::-1]   
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    rows = np.int_(img.shape[0]*freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
    
    return dom_patch

def describelabeledframes(config_path, output = True, plot = True):
    '''
    Describes dlc dataset by color distribution, size, intensity, and animal scale
    '''
    src = os.path.join(deeplabcut.auxiliaryfunctions.read_config(config_path)['project_path'], 'labeled-data')

    # scrap labels
    label_files = utils.scrapdirbystring(src, 'h5', output=False)
    dfs  = [pd.read_hdf(df) for df in label_files]
    labels = pd.concat(dfs, ignore_index=False)
    scorer = list(set(labels.columns.get_level_values('scorer'))).pop()

    # calculate scale
    if output:
        print('Calculating scales from labeled data...')
    scales = getboundingbox(labels)[scorer, 'boundingbox', 'scale']
    
    # scrap frames
    frames = utils.scrapdirbystring(src, 'png', output=False)

    # calculate frame sizes and intensity
    if output:
        print('Calculating frame sizes and intenisties...')
    dims = []
    intensities = []
    for frame in frames:
        img = cv2.imread(frame)
        dims.append(f'{img.shape[1]} x {img.shape[0]}')
        _, _, int = utils.get_avrg_color(img)
        intensities.append(int)

    # calculate dominant colors
    if output:
        print('Calculating dominant colors from labeled frames...')
    # PLACEHOLDER TODO
    #dom_patches = [calculatedominantcolors(cv2.imread(frame)) for frame in frames]
    dom_patch = calculatedominantcolors(cv2.imread(frames[10]))

    # plot descriptives
    if plot:
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15,10))
        ax0.hist(scales, bins = 100);
        ax0.set_title('Frame Scale in square pixel')
        ax1.hist(dims, bins = 10);
        ax1.set_title('Frame Sizes in pixel')
        ax2.imshow(dom_patch)
        ax2.set_title('Dominant colors')
        ax2.axis('off')
        ax3.hist(intensities, bins = 100);
        ax3.set_title('Frame intensities')
        plt.show()

    return

def extractframesfromsubset(config_path, subset, mode='automatic', algo='uniform', userfeedback=False, crop=False):
    '''
    Extract frames only from subset of video sets
    '''
    from pathlib import Path

    # save original video_set
    full_video_sets = deeplabcut.auxiliaryfunctions.read_config(config_path)['video_sets']

    # create subset of video_sets
    videos_subset = [os.path.join(os.path.dirname(config_path), "videos", os.path.basename(file)) for file in subset] 
    video_subsets = {}
    for video in videos_subset:
        rel_video_path = str(Path.resolve(Path(video)))
        vid = deeplabcut.utils.auxfun_videos.VideoReader(rel_video_path)
        video_subsets[rel_video_path] = {"crop": ", ".join(map(str, vid.get_bbox()))}

    edits = {'video_sets': video_subsets}
    deeplabcut.auxiliaryfunctions.edit_config(config_path, edits);

    # extract frames
    deeplabcut.extract_frames(config_path, mode=mode, algo=algo, userfeedback=userfeedback, crop=crop)
    
    # recover original video sets in config path
    edits = {'video_sets': full_video_sets}
    deeplabcut.auxiliaryfunctions.edit_config(config_path, edits);

    return

def monkeypatch():
    '''
    Patches the DLC modelzoo parameters to include URls to own models
    ATTENTION: Only works with DLC < 2.3
    '''
    # locate the `pretrained_model_urls.yaml` file in your local DeepLabCut installation (hardcoded based on deeplabcut dir structure)
    neturls_path = os.path.join(auxiliaryfunctions.get_deeplabcut_path(), 'pose_estimation_tensorflow', 'models', 'pretrained', 'pretrained_model_urls.yaml')

    # PLACEHOLDER FOR THE LAST ITERATION OF THE PIGEONSUPERMODEL
    edits = {'PigeonSuperModel_effnet_b0':  'https://gitlab.ruhr-uni-bochum.de/hidalggc/3dposetrackingoffreelymovingpigeons/-/raw/main/models/DLC_PigeonSuperModel_imgaug_efficientnet-b0.tar.gz',
             'RefModel_HexArena_resnet_50': 'https://gitlab.ruhr-uni-bochum.de/hidalggc/3dposetrackingoffreelymovingpigeons/-/raw/main/models/DLC_RefModel_HexArena_resnet_50.tar.gz', 
             'RefModel_ImagingSkinnerbox__resnet_50': 'https://gitlab.ruhr-uni-bochum.de/hidalggc/3dposetrackingoffreelymovingpigeons/-/raw/main/models/DLC_RefModel_ImagingSkinnerbox_resnet_50.tar.gz', 
             'RefModel_SkinnerBox_resnet_50': 'https://gitlab.ruhr-uni-bochum.de/hidalggc/3dposetrackingoffreelymovingpigeons/-/raw/main/models/DLC_RefModel_SkinnerBox_resnet_50.tar.gz',
             }
    
    # add new model to the neturls file     
    deeplabcut.auxiliaryfunctions.edit_config(neturls_path, edits)

    # add new model to locally loaded Modeloptions from modelzoo.py
    modelzoo.Modeloptions.extend(['PigeonSuperModel_effnet_b0', 'RefModel_HexArena_resnet_50', 'RefModel_ImagingSkinnerbox__resnet_50', 'RefModel_SkinnerBox_resnet_50'])
    
    return


def analyze_inference_speed(config_path):
    '''
    Returns a data frame with inference speeds of videos scrapped in the project directory
    '''
    pickle_files = [file for file in  utils.scrapdirbystring(os.path.dirname(config_path), '.pickle') if not 'training-datasets' in file]
    rows = []
    for metadata in pickle_files:
        mdata = pd.read_pickle(metadata)
        duration = mdata['data']['run_duration']
        dur_min = int(mdata['data']['run_duration']//60)
        dur_sec = int(mdata['data']['run_duration'] % 60)
        num_frames = int(mdata['data']['nframes'])
        speed = format(num_frames/duration, '.2f')
        dim_frames = str(mdata['data']['frame_dimensions'])
        network = str(mdata['data']['DLC-model-config file']['net_type'])
        video = os.path.basename(metadata).split('DLC_')[0]
        shuffle = os.path.basename(metadata).split('shuffle')[1].split('_')[0]
        iteration = os.path.basename(metadata).split('shuffle')[1].split('_')[1]
        rows.append([video, num_frames, dim_frames, shuffle, network, iteration, speed, str(dur_min).zfill(2)+':'+str(dur_sec).zfill(2)])
    
    inference_speed = pd.DataFrame(rows, columns=['video', 'frames', 'dimensions', 'shuffle', 'network', 'iteration', 'speed', 'duration'])
    
    return inference_speed

def plottracescomparison(videopath, searchlist, coordinate ='x', sortIndex =[]):
    '''
    Plots traces from a list searched by strings in model and video 
    TODO optimize search list 
    TODO add titles
    '''
    files = utils.scrapdirbystring(videopath, '.h5', output = False)
    for string in searchlist:
        files = [file for file in files if string in file]

    # sort files
    if len(sortIndex) > 0:
        files = [files[i] for i in sortIndex]
    else:
        pass
    
    fig, axes = plt.subplots(len(files),1, figsize = (20, 3*len(files)))

    for i, ax in enumerate(axes):
        # read data
        df = pd.read_hdf(files[i])
        filename = files[i].split(os.path.sep)[-2] +'_'+ files[i].split(os.path.sep)[-1].split('DLC')[0]

        # subset data
        xcols = [col for col in df.columns if coordinate in col]
        xcoord = df[xcols]

        # plot traces
        ax.plot(xcoord)
        if not i == 0:
            ax.set(ylabel = f'{coordinate}-coord in pixel', title=filename)
        else:
            ax.set(ylabel = f'{coordinate}-coord in pixel', title=f'{searchlist} \n \n{filename}')
            
    # set general parameters
    axes[i].set(xlabel = 'time in frames');
    plt.tight_layout();

    return files

def plotmodelevaluation(title, dfs, pcutoff, pck_threshold, ylim1=False,ylim2=False,ylim3=False,ylim4=False,ylim5=False,ylim6=False,):
    '''
    Plot evaluation resutls from a list of dfs generated by the function `aggregatepixelerror`
    '''
    
    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2,3, figsize =(2*10,1*10))
    plt.suptitle(title)

    ax1.set(ylabel = 'Likelihood [%]', xlabel = 'training iteration')
    if ylim1:
        ax1.set(ylim = ylim1)
    for df in dfs:
        ax1.plot(df['Likelihood'], label = df['model'].values[0])
    
    ax2.set(ylabel = f'PLK above pcutoff {pcutoff} [%]', xlabel = 'training iteration')
    if ylim2:
        ax2.set(ylim = ylim2)
    for df in dfs:
        ax2.plot(df['PLK'], label = df['model'].values[0])

    ax3.set(ylabel = f'PCK below {pck_threshold} error [%]', xlabel = 'training iteration')
    if ylim3:
        ax3.set(ylim = ylim3)
    for df in dfs:
        ax3.plot(df['PCK'], label = df['model'].values[0])

    ax4.set(ylabel = 'Tracking confidence', xlabel = 'training iteration')
    if ylim4:
        ax4.set(ylim = ylim4)
    for df in dfs:
        ax4.plot(df['conf'], label = df['model'].values[0])
    
    ax5.set(ylabel = 'RSME [pixel ± 2 SE]', xlabel = 'training iteration')
    if ylim5:
        ax5.set(ylim = ylim5)
    for df in dfs:
        ax5.plot(df['RSME'], label = df['model'].values[0])
        ax5.fill_between(df.index, df['RSME'] - 2* df['RSME_SEM'], df['RSME'] + 2* df['RSME_SEM'], alpha=0.2)

    ax6.set(ylabel = 'norm_RSME [scale ± 2 SE]', xlabel = 'training iteration')
    if ylim6:
        ax6.set(ylim = ylim6)
    for df in dfs:
        ax6.plot(df['norm_RSME'], label = df['model'].values[0])
        ax6.fill_between(df.index, df['norm_RSME'] - 2* df['norm_RSME_SEM'], df['norm_RSME'] + 2* df['norm_RSME_SEM'], alpha=0.2)
    
    plt.tight_layout;
    handles, label = ax1.get_legend_handles_labels()
    plt.legend(handles, label, bbox_to_anchor=(1.02, 2.2), loc='lower right', ncol=int(len(dfs)/2))

    plt.show()
    return

def comparelikelihoodviolines(file1, file2):
    '''
    Returns a violin plot comparing tracking likelihoods from two files at a time
    '''
    # assign model- and filenames
    if 'resnet' in file1:
        file1_name = "resnet"
    elif 'effnet' in file1:
        file1_name = "effnet"
    elif 'mobnet' in file1:
        file1_name = "mobnet"
    if 'resnet' in file2:
        file2_name = "resnet"
    elif 'effnet' in file2:
        file2_name = "effnet"
    elif 'mobnet' in file2:
        file2_name = "mobnet"

    if 'imgaug' in file1:
        file1_name = file1_name + '_imgaug'
    if 'imgaug' in file2:
        file2_name = file2_name + '_imgaug'
    
    if 'Foraging' in file1:
        video = "Foraging Platforms"
    elif 'Imaging' in file1:
        video = "Imaging Skinnerbox"
    elif '3DPOP' in file1:
        video = "3DPOP"

    data1 = pd.read_hdf(file1)
    data2 = pd.read_hdf(file2)

    likelihoods1 = pd.DataFrame()
    likelihoods2 = pd.DataFrame()

    for i, part in zip(range(len(data1.columns.levels[1])), list(data1.columns.levels[1])):
        likelihoods1[part] = data1[(data1.columns.levels[0][0], data1.columns.levels[1][i], data1.columns.levels[2][0])];
        likelihoods2[part] = data2[(data2.columns.levels[0][0], data2.columns.levels[1][i], data2.columns.levels[2][0])];

    sorted_likelihoods1 = likelihoods1[likelihoods1.median().sort_values().index]
    sorted_likelihoods2 = likelihoods2[likelihoods1.median().sort_values().index] # sort second as first

    fig, ax = plt.subplots(1,1, figsize = (7, 21))
    v1 = ax.violinplot(sorted_likelihoods1, vert=False, showmedians=True, showextrema=False, widths=0.9);
    # split violins
    for b in v1['bodies']:
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 1])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, m +0.45)
        b.set_color('b')
    v1['cmedians'].set_color('b')

    v2 = ax.violinplot(sorted_likelihoods2, vert=False, showmedians=True, showextrema=False, widths=0.9);
    # split violins
    for b in v2['bodies']:
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 1])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m-0.45, m)
        b.set_color('r')
    v2['cmedians'].set_color('r')

    ax.set_yticks(np.arange(1,len(sorted_likelihoods1.columns)+1), labels = list(sorted_likelihoods1.columns));
    ax2=ax.twiny()
    ax.set_xlabel('Prediction Likelihood')
    ax2.set_xlabel('Prediction Likelihood')

    ax.legend([v1['bodies'][0],v2['bodies'][0]],[file1_name, file2_name]);

    plt.title(video);
    plt.tight_layout()
    return


def read_training_logs(config_path):
    '''
    Returns a DataFrame with all training logs scraps in the projectpath provided
    '''
    projectpath = os.path.dirname(config_path)
    logs = utils.scrapdirbystring(projectpath, 'log.txt', output = False)
    dfs = []
    for logfile in logs:
        file = open(logfile, "r").read().split('\n')
        file = [line for line in file if 'iteration' in line]
        file = [line for line in file if 'loss' in line]
        file = [line for line in file if 'lr' in line]
        model = []
        elapsed = []
        iterations = []
        losses = []
        lrs = []
        oldtime = '00:00:00'
        # extract training statistics
        for step in file:
            try:
                info = step.split()
                it_idx = info.index('iteration:') +1
                loss_idx = info.index('loss:') +1
                lr_idx = info.index('lr:') +1

                newtime = info[1]
                iter = int(info[it_idx])
                secs = utils.elapsed_time(oldtime, newtime)
                loss = float(info[loss_idx])
                lr = float(info[lr_idx])
                oldtime = newtime

                # append lists
                model.append(logfile.split(os.path.sep)[-3])
                iterations.append(iter)
                elapsed.append(secs)
                losses.append(loss)
                lrs.append(lr)
            except:
                secs = iter = loss = lr = 0


        df = pd.DataFrame(list(zip(model, iterations, elapsed, losses, lrs)), columns = ['model', 'iteration', 'elapsed', 'loss', 'lr'])
        dfs.append(df)

    return pd.concat(dfs)



def getpixelerrors(config_path, output = True, debug = False):
    '''
    Returns and saves a data frame with the pixel error and tracking likelihoods
    '''

    # ignore warnings
    warnings.simplefilter('ignore')

    # search if pixelerror already exists
    projectpath = os.path.dirname(config_path)
    if debug:
        errors = ''
    else:
        errors = utils.scrapdirbystring(projectpath, 'pixelerrors.h5', output = False)
    if len(errors) > 0:
        print(f'Existing pixelerrors file found in {errors[0]}')
        pixelerrors = pd.read_hdf(errors[0])
    else:
        # evaluation data from deeplabcut.evaluate_network
        evals = [eval for eval in utils.scrapdirbystring(projectpath, '.h5', output = False) if 'evaluation-results' in eval]
        print(f'Calculating pixelerrors for {len(evals)} files found in {os.path.join(projectpath, "evaluation-results")}...')
        if debug:
            evals = evals[0:1]
        else:
            pass
        
        # manual labels as ground truth from deeplabcut.label_data
        labels = [label for label in utils.scrapdirbystring(projectpath, '.h5', output = False) if 'training-datasets' in label]
        manuallabels = pd.concat([pd.read_hdf(labels[i]) for i in range(len(labels))])

        # save pixel differences and tracking likelihoods
        pixelerrorlist = []
        # loop over evaluated snapshots
        for evaluation in evals:
            eval_df = pd.read_hdf(evaluation)
            snapshot, iteration = os.path.splitext(evaluation)[0].split('DLC_')[1].split('-snapshot-')
            model = snapshot.split('_'+iteration)[0]

            if output:
                print(f'Processing snapshot: {snapshot}')

            # loop over evaluated frames in snapshot
            for index, row in eval_df.iterrows():
                eval_row = eval_df.loc[index].droplevel(level = [0,1], axis = 0).droplevel(level = [0], axis = 1).head(1)

                # match evaluation and manual labels by index
                try:
                    label_row = manuallabels.loc[index].droplevel(level = [0,1], axis = 0).droplevel(level = [0], axis = 1).head(1)
                except:
                    # try different index, see imgaug
                    index = (index[0], 'original_' + index[1], index[2])
                    label_row = manuallabels.loc[index].droplevel(level = [0,1], axis = 0).droplevel(level = [0], axis = 1).head(1)
                frame = os.path.sep.join(index)

                # get likelihoods
                likelihoods = eval_row.xs('likelihood', axis = 1, level=1, drop_level=True).values

                # calculate pixel difference
                eval_row = eval_row.drop('likelihood', axis = 1, level = 1)
                pixeldiff = label_row - eval_row # no abs needed because this is squared later

                # calculate euclidean distance
                rse = np.sqrt(pixeldiff.xs('x', level = 1, axis = 1)**2 + pixeldiff.xs('y', level = 1, axis = 1)**2).values[0]
                
                # nan errors
                nanerrors = np.sum(np.sum(pixeldiff.isna()))/(np.sum(np.sum(pixeldiff.isna())) - np.sum(np.sum(np.isnan(rse))))

                # overwrite data
                pixeldiff.rename(columns={'x':'rse', 'y':'likelihood'}, inplace = True)
                pixeldiff.columns.rename('tracking', level = 1, inplace = True)
                pixeldiff.loc[:, pixeldiff.columns.get_level_values('tracking') == 'rse'] = rse
                pixeldiff.loc[:, pixeldiff.columns.get_level_values('tracking') == 'likelihood'] = likelihoods

                # add metadata
                pixeldiff.loc[:,'frame'] = frame
                pixeldiff.loc[:,'iteration'] = iteration
                pixeldiff.loc[:,'model'] = model

                # calculate scale as bounding box from manual labels
                xscale = (np.nanmin(label_row.xs('x', axis = 1, level=1, drop_level=False).values), 
                        np.nanmax(label_row.xs('x', axis = 1, level=1, drop_level=False).values))
                yscale = (np.nanmin(label_row.xs('y', axis = 1, level=1, drop_level=False).values), 
                        np.nanmax(label_row.xs('y', axis = 1, level=1, drop_level=False).values))
                scale = np.sqrt(abs(xscale[1] - xscale[0]) * abs(yscale[1] - yscale[0]))
                pixeldiff.loc[:,'scale'] = scale

                # save results for each frame
                pixelerrorlist.append(pixeldiff)

        pixelerrors = pd.concat(pixelerrorlist, axis = 0, ignore_index=True, copy=False)

        # save data
        if not debug:
            filename = os.path.join(projectpath, 'pixelerrors.h5')
            pixelerrors.to_hdf(filename, key = "df_with_missing", mode="w")
        else:
            pass

    # reset warning
    warnings.resetwarnings()

    return pixelerrors

def gettrainsplitfrompickle(pickle):
    '''
    Returns a list of frames used for training (training-split)
    '''
    metadata = pd.read_pickle(pickle)
    trainingset = []
    for img in metadata[0]:
        trainingset.append(os.path.sep.join(img['image']))
    return trainingset

def RMSEpixelerror(pixelerrors):
    '''
    Calculate RSME pixel error
    Note that rse error as euclidean distance is the root of sum of squares and not yet RMSE, see https://github.com/DeepLabCut/DeepLabCut/issues/60
    '''
    rse = pixelerrors.xs('rse', axis = 1, level=1, drop_level=True)
    norm_rse = rse.divide(pixelerrors['scale'], axis = 0)

    # calculate RSME for each keypoint over all frames, then aggregate over keypoints
    RSME = np.nanmean(np.nanmean(rse, axis = 0))
    
    # normalized RSE error by scale, then calculate RSME for each keypoint over frames and aggregate
    norm_RSME = np.nanmean(np.nanmean(norm_rse, axis = 0))

    # calculate SE between frames (=> average consistency)
    std = np.nanstd(rse, axis =0)
    RSME_SE = np.nanmean(std) / np.sqrt(sum(np.isfinite(std)))
    std = np.nanstd(norm_rse, axis =0)
    norm_RSME_SE = np.nanmean(std) / np.sqrt(sum(np.isfinite(std)))

    # calculate SE between keypoints (=> average homogeneity)
    std = np.nanstd(rse, axis =1)
    RSME_SE_keypoint = np.nanmean(std) / np.sqrt(sum(np.isfinite(std)))
    std = np.nanstd(norm_rse, axis =1)
    norm_RSME_SE_keypoint = np.nanmean(std) / np.sqrt(sum(np.isfinite(std)))
    
    return RSME, norm_RSME, RSME_SE, norm_RSME_SE, RSME_SE_keypoint, norm_RSME_SE_keypoint


def aggregatepixelerror(pixelerrors, split, bodyparts, pcutoff = 0.5, pck_threshold = 0.33, filter_pcutoff = False, output = False):
    '''
    Returns pixel errors as RSM, norm_RSM and PCK for the passed data split 
    '''
    # ignore warnings
    warnings.simplefilter('ignore')
    
    # subset by split
    subset_pixelerrors = pixelerrors[pixelerrors['frame'].isin(split)].reset_index(drop=True)

    # TODO subset by bodyparts

    # calculate evaluation errors
    evaluation = []
    for model in set(subset_pixelerrors['model']):
        model_pixelerrors = subset_pixelerrors[subset_pixelerrors['model'] == model].reset_index(drop=True)

        for iteration in set(model_pixelerrors['iteration']):
            iter_pixelerrors = model_pixelerrors[model_pixelerrors['iteration'] == iteration].reset_index(drop=True)

            # calculate likelihoods
            likelihood =  np.mean(np.nanmean(iter_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True), axis = 0,))

            # filter by pcutoff
            if filter_pcutoff:
                filter_pcutoff = pcutoff
            else:
                filter_pcutoff = 0
            pcutoff_pixelerrors = iter_pixelerrors[iter_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True)>=filter_pcutoff]
            pcutoff_pixelerrors['frame'] = iter_pixelerrors['frame']
            pcutoff_pixelerrors['iteration'] = iter_pixelerrors['iteration']
            pcutoff_pixelerrors['model'] = iter_pixelerrors['model']
            pcutoff_pixelerrors['scale'] = iter_pixelerrors['scale']

            # calculate errors and SEM
            RSME, norm_RSME, RSME_SE, norm_RSME_SE, RSME_SE_keypoint, norm_RSME_SE_keypoint = RMSEpixelerror(pcutoff_pixelerrors)

            # calulate PCK and PLK
            PLK = np.nanmean(pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True)[pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True)>=pcutoff].count(axis=0).values/pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True).notna().sum())
            normalized_error = pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True).subtract(pcutoff_pixelerrors.xs('scale', axis = 1,)*pck_threshold, axis = 0)
            PCK = np.nanmean(pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True)[normalized_error<=0].count(axis=0).values/pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True).notna().sum()) 


            # calculate tracking confidence
            likely = np.nanmean(pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True), axis = 0)
            error = np.nanmean(pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True).divide(pcutoff_pixelerrors['scale'], axis = 0), axis = 0)
            notnan_idx = np.isfinite(likely) & np.isfinite(error)
            conf = np.corrcoef(likely[notnan_idx], error[notnan_idx], rowvar = False)[0][1]

            # save as dataframe TODO rename columns
            data = {'model': model, 'iteration': int(iteration), 'RSME': RSME, 'RSME_SEM': RSME_SE, 'RSME_SEM_keypoint': RSME_SE_keypoint, 'norm_RSME': norm_RSME, 'norm_RSME_SEM':norm_RSME_SE, 'norm_RSME_SEM_keypoint':norm_RSME_SE_keypoint, 'PCK': PCK, 'Likelihood': likelihood, 'PLK': PLK,  'conf': conf}
            evaluation.append(data)
            
            if output:
                print(f'pixel error for {model}, it: {iteration}, RSME: {RSME}, norm_RSME: {norm_RSME}, Likelihood: {likelihood}, confidence; {conf}')
            
    # reset warning
    warnings.resetwarnings()
    
    df = pd.DataFrame(data = evaluation, index = [i for i in range(len(evaluation))])

    return df

def perkeypointpixelerror(pixelerrors, split,  pcutoff = 0.5, pck_threshold = 0.33, filter_pcutoff = False, output = False):
    '''
    Returns the average pixel errors and likelihoods for each individual keypoint
    '''
    # ignore warnings
    warnings.simplefilter('ignore')
    
    # subset by split
    subset_pixelerrors = pixelerrors[pixelerrors['frame'].isin(split)].reset_index(drop=True)

    # calculate evaluation errors
    evaluation = []
    for model in set(subset_pixelerrors['model']):
        model_pixelerrors = subset_pixelerrors[subset_pixelerrors['model'] == model].reset_index(drop=True)

        for iteration in set(model_pixelerrors['iteration']):
            iter_pixelerrors = model_pixelerrors[model_pixelerrors['iteration'] == iteration].reset_index(drop=True)

            # calculate likelihood
            likelihood = np.nanmean(iter_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True), axis = 0)
            std = np.nanstd(iter_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True), axis = 0,)
            likelihood_SE = std / np.sqrt(sum(np.isfinite(std)))

            # filter by pcutoff
            if filter_pcutoff:
                filter_pcutoff = pcutoff
            else:
                filter_pcutoff = 0
            pcutoff_pixelerrors = iter_pixelerrors[iter_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True)>=filter_pcutoff]
            pcutoff_pixelerrors['frame'] = iter_pixelerrors['frame']
            pcutoff_pixelerrors['iteration'] = iter_pixelerrors['iteration']
            pcutoff_pixelerrors['model'] = iter_pixelerrors['model']
            pcutoff_pixelerrors['scale'] = iter_pixelerrors['scale']

            # calculate pcutoff_likelihood
            pcutoff_likelihood = np.nanmean(pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True), axis = 0)
            std = np.nanstd(pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True), axis = 0,)
            pcutoff_likelihood_SEM = std / np.sqrt(sum(np.isfinite(std)))

            # calculate errors
            RSME = np.nanmean(pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True), axis = 0)
            std = np.nanstd(pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True), axis = 0)
            RSME_SE = std / np.sqrt(sum(np.isfinite(std)))
            norm_RSME = np.nanmean(pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True).divide(pcutoff_pixelerrors['scale'], axis = 0), axis = 0)
            std = np.nanstd(pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True).divide(pcutoff_pixelerrors['scale'], axis = 0), axis = 0)
            norm_RSME_SE = std / np.sqrt(sum(np.isfinite(std)))

            # PLK and PCK
            PLK = pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True)[pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True)>=pcutoff].count(axis=0).values/len(pcutoff_pixelerrors)
            normalized_error = pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True).subtract(pcutoff_pixelerrors.xs('scale', axis = 1,)*pck_threshold, axis = 0)
            PCK = pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True)[normalized_error<=0].count(axis=0).values/len(pcutoff_pixelerrors)

            # calculate confidence
            likely = pcutoff_pixelerrors.xs('likelihood', axis = 1, level=1, drop_level=True)
            error = pcutoff_pixelerrors.xs('rse', axis = 1, level=1, drop_level=True).divide(pcutoff_pixelerrors['scale'], axis = 0)
            notnan_idx = np.isfinite(likely) & np.isfinite(error)
            conf = likely[notnan_idx].corrwith(error[notnan_idx], axis = 0)

            # save data as dict
            data = {'RSME': RSME, 'RSME_SE': RSME_SE, 'norm_RSME':norm_RSME, 'norm_RSME_SE':norm_RSME_SE, 'Likelihood': likelihood,'Likelihood_SE': likelihood_SE, 'PCK': PCK, 'PLK': PLK, 'conf': conf}
            
            # generate multiindex for keypoints
            df = pd.DataFrame(data).stack().to_frame().T#.reset_index()
            df.columns.set_levels(list(pd.unique(pcutoff_pixelerrors.columns.get_level_values(0))),level=0,inplace=True)
            df.index = pd.MultiIndex.from_tuples([(model, int(iteration))], names=('model', 'iteration'))
            evaluation.append(df)

    df = pd.concat(evaluation).sort_index()

    return df

def labelstablecoordinates(video, df_size = 1, save = False):
    '''
    Returns a df in DLC format for manually labeled reference coordinates
    '''

    def keyboard_input(x, y, font, color):
        text = ""
        letters = string.ascii_letters + string.digits + string.punctuation
        while True:
            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    text = text + letter
                    # refresh image with text
                    cv2.putText(image, text, (x,y), font, 0.5, color, 1)
                    cv2.imshow('Frame', image)
            if key == ord("\n") or key == ord("\r"): # Enter Key
                break
        return text

    def click_event(event, x, y, flags, coords):
        
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:

            # displaying the marker with coordinates
            markerSize = 15
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            color2 = (0, 255, 0)
            
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, markerSize, thickness)
            cv2.imshow('Frame', image)

            # ask for label
            text = keyboard_input(x, y, font, color)
            coords[text] = {'x': x, 'y':y, 'likelihood': 0.99}

            # refresh with text
            cv2.drawMarker(image, (x, y), color2, cv2.MARKER_CROSS, markerSize, thickness)
            cv2.imshow('Frame', image)

    def correctpoint(event, x, y, flags, coords):
        pass
        return
    
    videocap = cv2.VideoCapture(video)
    numFrames = videocap.get(cv2.CAP_PROP_FRAME_COUNT)
    randFrames = np.sort(np.random.randint(numFrames, size = (5)))
    for frame in randFrames[0:1]:
        videocap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        success, image = videocap.read()
        image = cv2.putText(image, f'frame:{frame}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), )
        if success:
            # print instructions
            help1 = '1. label stable points with left mouse click'
            help2 = '2. label point name with keyboard'
            help3 = '3. confirm each label with enter, press any Key to exit'
            cv2.putText(image, help1, (50,100), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), )
            cv2.putText(image, help2, (50,120), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), )
            cv2.putText(image, help3, (50,140), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), )
            cv2.imshow('Frame', image)
            coords = dict()
            cv2.setMouseCallback('Frame', click_event, coords)
            cv2.waitKey(0)
            
    # check point stability over other frames
    for frame in randFrames[1:]:
        videocap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        success, image = videocap.read()
        image = cv2.putText(image, f'frame:{frame}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), )
        if success:
            # print instructions
            help1 = 'confirm points are stable across frames, press enter'
            cv2.putText(image, help1, (50,100), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), )
            cv2.setMouseCallback('Frame', correctpoint)
            # print coords in image
            for p in coords.keys():
                label = p
                x = coords[p]['x']
                y = coords[p]['y']
                cv2.drawMarker(image, (x, y), (255,255,255), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(image, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), )
            cv2.imshow('Frame', image)

            cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save coords in DLC format
    reformed_dict = {}
    for outerKey, innerDict in coords.items():
        for innerKey, values in innerDict.items():
            reformed_dict[('ManualReference', outerKey, innerKey)] = values
    refs = pd.DataFrame(reformed_dict, index = range(df_size))
    refs.columns.set_names(['scorer', 'bodyparts', 'coords'], inplace=True)

    if save:    
        # save as hdf
        filename = os.path.splitext(video)[0]+'.hdf'
        refs.to_hdf(filename, key = "df_with_missing", mode="w")
    else:
        pass

    return refs
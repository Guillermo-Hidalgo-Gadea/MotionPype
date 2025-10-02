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
import ffmpeg
import shutil
import datetime
import numpy as np
import pandas as pd
from urllib import request

from pathlib import Path
import matplotlib.pyplot as plt



def scrapdirbystring(dir, query, output=True):
    '''
    Return list of scrapped file paths that match querry string. 
    '''
    scrapped_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if query in os.path.join(root, file):
                scrapped_files.append(os.path.join(root, file))

    if output:
        print(f'Scrapping files in {dir} returned list of size: {len(scrapped_files)}')

    return scrapped_files

def scrapfoldersindir(dir, output=True):
    '''
    Return list of scrapped folder paths in directory. 
    '''
    scrapped = []
    for root, dirs, files in os.walk(dir):
        scrapped.append(root)
        for dir in dirs:
            scrapped.append(os.path.join(root, dir))
        for file in files:
            scrapped.append(os.path.join(root))

    if output:
        print(f'Scrapping in {dir} returned list of size: {len(list(set(scrapped)))}')

    return list(set(scrapped))

def batchrenamebyrules(files, dest, extension, projectname, keyword, cam_assignement, rule1, rule2, rule3, rename = False):
    '''
    Move and batch rename files by specified filenaming rules. 
    Current naming convention "YYYYMMDD_ProjectName_PIDS_keyword_camA.avi"
    
    # TODO define and generalize naming rules
    # rule 1: separator '_' used for filenames and '\\' for paths
    # rule 2: name always contains date and time 
    # rule 3: PIDS can be read from the directory in form _PXXX_
    # camera assignement: 8 digit serial number for camera assignement
    '''
    if not rename:
        print('Please doublecheck the following renaming criteria:')
        print(f'Rule 1 - Separators: {rule1}')
        print(f'Rule 2 - Date: {rule2}')
        print(f'Rule 3 - File ID: {rule3}')
        print(f'Cam assignement: {cam_assignement}')

    for file in files:
        passed = False
        # rule1: split filename by separatot
        components = re.split(rule1, file)
        # rule2: extract date from filename
        date = [component for component in components if rule2 in component].pop()
        # rule3: extract file id from filename
        pid = [component for component in components if rule3 in component].pop()
        # camera assignement by serial or cam id
        serial = [component for component in components if component in cam_assignement.keys()].pop() if [component for component in components if component in cam_assignement.keys()] else False
        cam = [component for component in components if component in cam_assignement.values()].pop() if [component for component in components if component in cam_assignement.values()] else False
        
        if serial:
            camera = cam_assignement[serial]
        elif cam:
            camera = cam
        else:
            print(f'Error assigning camera ID for {file}')
            passed = True

        if not passed:
            # filename convention
            ordered = date +'_'+ projectname +'_'+ pid +'_'+ keyword +'_'+ camera + extension
            # rename file
            renamed = os.path.join(dest, ordered)
            print(f'Renaming file {file} into to {renamed}')
            if rename:
                os.rename(os.path.join(file), os.path.join(renamed))
    return

def comparedirectories(dir1, dir2, separator):
    '''
    Return a dataframe matching content of two directories with file number and content size.
    '''
    matches = pd.DataFrame([], columns=[f'{dir1}', 'numfiles', 'size', f'{dir2}', 'numfiles', 'size'])
    unique = []
    errors = []

    # for each file in dir1 loop over all files in dir2
    for file1 in os.listdir(dir1):
        for file2 in os.listdir(dir2):
            try:
                # strip filenames in parts
                splits1 = file1.split(separator)
                splits2 = file2.split(separator)
                part1 = (splits1[0], splits2[0],)
                part2 = (splits1[-2], splits2[-2],)
                part3 = (splits1[-1], splits2[-1],)

                if part1[0] == part1[1]:
                    if part2[0] == part2[1]:
                        if part3[0] in part3[1] or part3[1] in part3[0]:
                            # get content
                            contentdir1 = len(os.listdir(os.path.join(dir1, file1)))
                            contentdir2 = len(os.listdir(os.path.join(dir2, file2)))
                            # get size
                            sizedir1 = "{:.2f}".format(sum([os.path.getsize(os.path.join(dir1, file1, f)) for f in os.listdir(os.path.join(dir1, file1))])* 9.31*10**(-10))
                            sizedir2 = "{:.2f}".format(sum([os.path.getsize(os.path.join(dir2, file2, f)) for f in os.listdir(os.path.join(dir2, file2))])* 9.31*10**(-10))
                            # write table
                            matches.loc[len(matches)] = [file1, contentdir1, sizedir1, file2, contentdir2, sizedir2]
                        else:
                            unique.append(file1)
                    else:
                        unique.append(file1)
                else:
                    unique.append(file1)
            except:
                errors.apend(file1)

    return matches, unique, errors

def listextensions(files, dir):
    '''
    Return list of extensions from file list or directory.
    '''
    if dir:
        files = scrapdirbystring(dir, '', output=False)

    files = [os.path.basename(file) for file in files]
    extensions = [file.split('.')[-1] for file in files]

    return set(extensions)

def creatreferencevideo(videos, destination, crf):
    '''
    Create shorter/fastforward reference videos from behavior videos.
    TODO
    '''
    for video in videos:
        output = os.path.join(destination, os.path.basename(video))
        extracted = os.path.join(destination, 'img%03d.jpg')
        extract_command = f'ffmpeg -i {video} -vf fps=1 {extracted}'
        stitch_command = f'ffmpeg -r 24 -i {extracted} -c:v libvpx-vp9 -b:v 2000k -crf {crf} {output}'
        
    ### Alternative from StopSignal
    # TODO
    # loop over sessions in project
    for session in os.listdir(projectpath):
        try: 

            # check if raw-videos exist
            if 'videos-raw' in os.listdir(os.path.join(projectpath, session)):

                # check if reference videos exist
                content_raw = os.listdir(os.path.join(projectpath,session,'videos-raw'))
                content_ref = os.listdir(os.path.join(projectpath,session,'reference'))

                if len(content_ref) == len(content_raw):
                    print('Reference videos already exist.')
                else:
                    # create reference video
                    print(f'Creating reference videos for session: {session}')

                    # get raw video
                    for rawvideo in content_raw:
                        video = os.path.join(projectpath, session, 'videos-raw', rawvideo) # extract frames, how many? which ones?
                        print(f'Processing: {video}')

                        # create frames folder
                        outpath = os.path.join(projectpath, session, 'reference-frames')
                        try:
                            os.makedirs(outpath)
                        except:
                            pass

                        # Read the video
                        input = cv2.VideoCapture(video)
                        
                        if not input.isOpened():
                            print('Error opening video.')
                        else:
                            # choose frames to select
                            frame_count = int(input.get(cv2.CAP_PROP_FRAME_COUNT))
                            selected_frames = np.linspace(0, frame_count-1, num=50, dtype = int) # how many frames to extract
                            print(f'Extracting {50} reference frames from num_frames: {frame_count}...')

                            # start reading frames
                            framecounter = 0

                            while True:
                                ret, frame = input.read()

                                if ret:
                                    if framecounter in selected_frames:
                                        cv2.imwrite(os.path.join(outpath, "frame{:d}.jpg".format(framecounter)), frame)
                                    else:
                                        pass
                                else:
                                    break

                                framecounter += 1
    
                        # Release all space and windows once done
                        input.release()

                        # create video from frames
                        print('Creating reference video...')
                        img_array = []
                        for frame in os.listdir(outpath):
                            img = cv2.imread(os.path.join(outpath,frame))
                            height, width, layers = img.shape
                            size = (width,height)
                            img_array.append(img)

                        filename = os.path.join(projectpath,session,'reference','reference_' + os.path.basename(video))
                        out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MP4V'), 5, size)
                        
                        for i in range(len(img_array)):
                            out.write(img_array[i])
                        out.release()
                        print(f'Reference Video Created: {filename}')

                        # remove reference frames and directory
                        print('Removing extracted reference frames...')
                        for frame in os.listdir(outpath):
                            os.remove(os.path.join(outpath,frame))
                        os.removedirs(outpath)

            else:
                pass
        except:
            print('oops')
            pass

    return

def extractvideoframes(file, frames_to_extract, framesize, outputdir, output=False):
    '''
    Extract first video frames for a given interval.
    Useful to match and synchronize video streams manually by frame number. 
    TODO
    ''' 
    # check output directory
    source = os.path.basename(file).split('.')[0]
    outpath = os.path.join(outputdir, source)
    if os.path.exists(outpath):
        pass
    else:
        os.makedirs(outpath)

    # get video file
    if os.path.exists(file):
        video = file
        if output:
            print(f'Processing: {video}')
        # Read video
        input = cv2.VideoCapture(video)
        if not input.isOpened():
            print('Error opening video.')
        else:
            if output:
                print(f'Extracting first {frames_to_extract} frames...')
            # start reading frames
            for i in range(frames_to_extract):
                ret, frame = input.read()
                if ret:
                    # downscale frame
                    frame = cv2.resize(frame, framesize)
                    cv2.imwrite(os.path.join(outpath, "frame_{:04d}.jpg".format(i)), frame)
                else:
                    pass

            input.release()
            if output:
                print(f'Frames extracted to: {outpath}')

    else:
        print(f'File {file} not found.')

    return

def synchronize_cameras(config_path, timesync, sessions):
    '''
    TODO
    see Stop Signal
    get timesync dict and corresponding pose files from project
    trim dfs with start frame
    '''
    # get config parameters
    configtoml = toml.load(config_path)
    pose_dir = configtoml['pipeline']['pose_2d_filter']
    output_dir = configtoml['pipeline']['pose_2d_sync']
    projectpath = configtoml['project']
    # loop over sessions
    for session in sessions:
        print(f'Processing session: {session}')
        try:
            datadir = os.path.join(projectpath, session, pose_dir)
            posedata = utils.scrapdirbystring(datadir, 'h5', output=False)
            output = os.path.join(projectpath, session, output_dir)

            # create output dir
            if os.path.isdir(output):
                pass
            else:
                os.makedirs(output)

            # loop over files
            for file in posedata:
                # read data
                df = pd.read_hdf(file)
                # find match
                match = session + '_' + file.split('_')[-1].split('.')[0]
                startframe = timesync[match]
                # trim df
                new_df = df[startframe:-1].reset_index(drop=True)
                # TODO trim end for same length

                # save new df
                outputfile = output + '/' + 'sync_' + os.path.basename(file)
                new_df.to_hdf(outputfile, key ='df', mode='w')
                print(f'File synchronized: {outputfile}')

        except:
            print(f'Error processing session: {session}')
    return

def create_dummy_videos(target_dir, url):
    '''
    Create dummy video from image in url
    Useful to create empty DLC projects, see errors with empty project.
    '''

    # download logo
    logo = os.path.join(target_dir, 'logo.png')
    request.urlretrieve(url, logo);
    # create dummy video
    dummyvideo = os.path.join(target_dir, 'logo.mp4')
    img_array = []
    for i in range(20):
        img = cv2.imread(logo)
        height, width, layers = img.shape
        img_array.append(img)
    
    out = cv2.VideoWriter(dummyvideo, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    os.remove(logo)

    return dummyvideo

def moveorcopyfiles(src, dst, query, move = True, output=False):
    '''
    Moves or copies files between directories
    '''
    filelist = []
    
    files = scrapdirbystring(src, query, output=False)
    new_files = [os.path.join(dst, os.path.relpath(file, src)) for file in files]

    for old, new in zip(files, new_files):
        if not os.path.exists(os.path.dirname(new)):
            os.makedirs(os.path.dirname(new))

        filelist.append(old)
        
        if move:
            action = 'moved'
            os.replace(old, new)
        else:
            action = 'copied'
            shutil.copyfile(old, new)

    if output:
        print(f'{len(filelist)} files {action} to {dst}.')

    return

def cvtracking():
    '''
    TODO see Kalman Filter animalposeplotter
    Tutorial from Rahmad Sadli: https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
    '''
    return


def elapsed_time(t1, t2):
    '''
    Return the absolute time difference in seconds from two times in string format
    "13:23:34" - "13:28:34" = 300
    '''
    t1 = datetime.datetime.strptime( t1, '%H:%M:%S').time()
    t2 = datetime.datetime.strptime( t2, '%H:%M:%S').time()
    t1_secs = t1.second + 60 * (t1.minute + 60* t1.hour)
    t2_secs = t2.second + 60 * (t2.minute + 60* t2.hour)

    return abs(t2_secs - t1_secs)


def get_video_metadata(videolist):
    '''
    Return a DataFrame with relevant video metadata from a list of video paths
    '''
    video_metadata = []
    for file in videolist:
        name = ('_').join(file.split(os.path.sep)[-1].split('_')[1:])
        filetype = file.split(os.path.sep)[-1].split('_')[0]
        MB = round(os.stat(file).st_size /1024 /1024,2)
        metadata  = ffmpeg.probe(file)["streams"][0]
        nb_frames = int(metadata["nb_frames"])
        fps = int(eval(metadata["avg_frame_rate"]))
        duration = int(float(metadata["duration"]))
        bitrate = int(metadata["bit_rate"])
        width = int(metadata["width"])
        height = int(metadata["height"])

        video_metadata.append( pd.DataFrame(
            data = {"type":filetype, "nb_frames":nb_frames, 
            "duration":duration, "fps":fps, "width":width, "height":height,"bitrate":bitrate, "MB":MB
            }, index = [name]))

    df = pd.concat(video_metadata).sort_index()
    df.index.names = ["video"]
    return df


def get_avrg_color(img):
    avrg_color = np.average(np.average(img, axis=0), axis=0)
    color = np.ones((img.shape[1],img.shape[0],3), dtype=np.uint8)
    color[:,:] = avrg_color

    redness = np.average(np.average(color, axis=0), axis=0)[-1]/(np.sum(np.average(np.average(color, axis=0), axis=0))+0.001)
    intensity = np.sum(np.average(np.average(color, axis=0), axis=0))/np.sum(np.array([255, 255, 255]))

    return color, redness, intensity

def get_gamma(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mid = 0.5
    mean = np.mean(val)
    gamma = np.log(mid*255)/np.log(mean)

    return gamma

def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    '''
    see here: https://stackoverflow.com/a/57046925
    '''
    # Calculate grayscale histogram
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    try:
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

    except:
        alpha = 1
        beta = 0    

    return alpha, beta

def adaptive_threshold_correction(img, blocksize, c):
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    threshold = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c) 

    return threshold

def contrast_brightness_gamma_correction(img, alpha, beta, gamma):
    '''
    This function performs contrast, brightness and gamma correction on a given frame
    see here: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    alpha = 1, beta = 0 and gamma = 1 leave the image unchanged
    '''
    
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    contrast_brightness = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    corrected = cv2.LUT(contrast_brightness, lookUpTable)

    return corrected

def color_correction(img):
    red = img.copy()
    red[:,:,(0,1)] = 0
    
    return red

def blur_correction(img, compression = 0.95):
    kernel = int(min(img.shape[1], img.shape[0]) * (1-compression))
    if kernel % 2 == 0:
        kernel +=1 # Even 
    else:
        pass # Odd
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)

    return blur

def compression_correction(frame, crf, outputdir):
    encode = f'ffmpeg -y -framerate 1 -i concat:{frame} -vcodec h264 -crf {crf} out.avi'
    os.system(encode)
    compressed_frame = os.path.join(os.path.abspath(outputdir), os.path.basename(os.path.dirname(frame))+'_compressed_'+os.path.basename(frame))
    # 'H:\\ImagingSkinnerbox_local\\ImEmbMem_ref-RW-2022-04-19\\compresed-data\\20220301_ImagingSkinnerbox_P407_cam003_compressed_img005280.png'
    decode = f'ffmpeg -y -i out.avi -r 1/1 {compressed_frame}'
    os.system(decode)
    
    compressed = cv2.imread(compressed_frame)
    os.remove(compressed_frame)

    return compressed

def scaling_correction(img, scale_percent=80):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    rescaled = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    downscaled = cv2.resize(rescaled, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)

    return downscaled

def sharpen(img):
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel_sharpen_edges = np.array([[-1,-1,-1,-1,-1],
                                     [-1,2,2,2,-1],
                                     [-1,2,8,2,-1],
                                     [-1,2,2,2,-1],
                                     [-1,-1,-1,-1,-1]]) / 8.0
    sharpened = cv2.filter2D(img, -1, kernel_sharpen_edges)
    return sharpened

def rescaleframe(img, rescale_factor):
    '''
    Returns a rescaled image
    '''
    width = int(img.shape[1] * rescale_factor)
    height = int(img.shape[0] * rescale_factor)
    rescaled = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    return rescaled

def rescaledlcframelabels(scalefactor, framesets, output=False):
    '''
    Rescale already labeled frames from DLC project
    '''

    for hdf in framesets:
        labels = pd.read_hdf(hdf)
        scorer = list(set(labels.columns.get_level_values('scorer')))[0]
        for bodypart in list(set(labels.columns.get_level_values('bodyparts'))):
            labels[scorer, bodypart] *= scalefactor
        labels.to_hdf(hdf, key = "df_with_missing", mode="w")
        
        if output:
            print(f'File {hdf} rescaled by {scalefactor}')
    return

def manualimageaugmentation(config_path, augmentations, intensity_threshold = 0.15, crf = 30, compression = 0.99, scale_percent=50, rescale_factor=0.33):
    '''
    Applies manual image augmentation from list
    '''
    import deeplabcut
    valid_aug = [
                'corrected', 
                'underexposed',
                'redfilterd',
                'blurred',
                'compressed',
                'downscaled',
                'rescaled',
                ]

    # read dataset from project
    dataset_path = os.path.join(deeplabcut.auxiliaryfunctions.read_config(config_path)['project_path'], 'labeled-data')
    
    # scrap labels and frames
    label_files = scrapdirbystring(dataset_path, 'h5', output=False)
    frames = scrapdirbystring(dataset_path, 'png', output=False)
    
    # create augmented dataset in new location for safety
    imgaug_dataset_path = os.path.join(deeplabcut.auxiliaryfunctions.read_config(config_path)['project_path'], 'augmented-data')
    if not os.path.exists(imgaug_dataset_path):
        os.makedirs(imgaug_dataset_path)

    for aug in augmentations:
        if aug not in valid_aug:
            print(f'Unknown augmentation method: {aug}')
            print(f'Check or update `valid_aug` parameter: {valid_aug}')
            break

        print(f"Running {aug} augmentation...")
        # read frames
        for frame in frames:
            # check image properties
            img =  cv2.imread(frame)
            alpha, beta = automatic_brightness_and_contrast(img, clip_hist_percent=10)
            color, redness, intensity = get_avrg_color(img)
            gamma = get_gamma(img)

            # create original img
            framename = "original_"+frame.split(dataset_path)[-1].strip(r"\\")
            newpath = os.path.join(imgaug_dataset_path, framename)
            if not os.path.exists(os.path.dirname(newpath)):
                os.makedirs(os.path.dirname(newpath))

            # skip if existing from previous loop
            if not os.path.exists(newpath):
                cv2.imwrite(newpath, img)

            if aug == 'corrected':

                if redness > 0.50:
                    # this is a red frame, gamma correction changes colorspace, keep it in grays
                    bw_img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
                    corrected = sharpen(contrast_brightness_gamma_correction(bw_img, alpha=alpha, beta=beta, gamma=gamma))
                else:
                    corrected = sharpen(contrast_brightness_gamma_correction(img, alpha=alpha, beta=beta, gamma=gamma))

                # corrected
                framename = "corrected_"+frame.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, framename)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                cv2.imwrite(newpath, corrected)

            elif aug == 'underexposed':
        
                if intensity < intensity_threshold:
                    # this is a very dark frame, be conservative with darkening
                    gamma = gamma/ 2
                    dark_gamma = gamma
                else: 
                    dark_gamma = 4-gamma
                dark = contrast_brightness_gamma_correction(img, alpha=1, beta=0, gamma= dark_gamma)
                
                # underexposed
                framename = "underexposed_"+frame.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, framename)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                cv2.imwrite(newpath, dark)

            elif aug =='redfilterd':
                
                red = color_correction(img)

                # redfilterd
                framename = "redfilterd_"+frame.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, framename)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                cv2.imwrite(newpath, red)

            elif aug =='blurred':
        
                blur = blur_correction(img, compression = compression)

                # blurred
                framename = "blurred_"+frame.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, framename)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                cv2.imwrite(newpath, blur)

            elif aug =='compressed':
                
                outputdir = os.path.join(deeplabcut.auxiliaryfunctions.read_config(config_path)['project_path'], 'compresed-data')
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                compressed = compression_correction(frame, crf=crf, outputdir=outputdir)
                
                # compressed
                framename = "compressed_"+frame.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, framename)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                cv2.imwrite(newpath, compressed)

            elif aug =='downscaled':

                downscaled = scaling_correction(img, scale_percent=scale_percent)

                # downscaled
                framename = "downscaled_"+frame.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, framename)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                cv2.imwrite(newpath, downscaled)

            elif aug =='rescaled':
            
                rescaled = rescaleframe(img, rescale_factor)

                # rescaled
                framename = "rescaled_"+frame.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, framename)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                cv2.imwrite(newpath, rescaled)
            
            else:
                pass
                # adaptive threshold, this is useful to check informatioon loss
                #threshold = cv2.cvtColor(adaptive_threshold_correction(dark, blocksize = 3, c=1),cv2.COLOR_GRAY2BGR)
                #original_th = cv2.cvtColor(adaptive_threshold_correction(img, blocksize = 3, c=1),cv2.COLOR_GRAY2BGR)

        # create synthetic labels
        for label in label_files:

            # save original
            apx = "original_"
            newname = apx+label.split(dataset_path)[-1].strip(r"\\")
            newpath = os.path.join(imgaug_dataset_path, newname)
            if not os.path.exists(os.path.dirname(newpath)):
                os.makedirs(os.path.dirname(newpath))

            # skip if existing from previous loop
            if not os.path.exists(newpath):
                # change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                shutil.copy(os.path.splitext(label)[0]+".csv", os.path.splitext(newpath)[0]+".csv")
            
            if aug == 'corrected':
                # corrected_
                apx = "corrected_"
                newname = apx+label.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, newname)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                ## change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                shutil.copy(os.path.splitext(label)[0]+".csv", os.path.splitext(newpath)[0]+".csv")

            elif aug == 'underexposed':
                # dark_
                apx = "underexposed_"
                newname = apx+label.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, newname)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                ## change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                shutil.copy(os.path.splitext(label)[0]+".csv", os.path.splitext(newpath)[0]+".csv")

            elif aug =='redfilterd':
                # red_
                apx = "redfilterd_"
                newname = apx+label.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, newname)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                ## change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                shutil.copy(os.path.splitext(label)[0]+".csv", os.path.splitext(newpath)[0]+".csv")

            elif aug =='blurred':
                # blur_
                apx = "blurred_"
                newname = apx+label.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, newname)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                ## change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                shutil.copy(os.path.splitext(label)[0]+".csv", os.path.splitext(newpath)[0]+".csv")

            elif aug =='compressed':
                # compressed_
                apx = "compressed_"
                newname = apx+label.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, newname)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                ## change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                shutil.copy(os.path.splitext(label)[0]+".csv", os.path.splitext(newpath)[0]+".csv")

            elif aug =='downscaled':
                # downscaled_
                apx = "downscaled_"
                newname = apx+label.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, newname)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                ## change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                shutil.copy(os.path.splitext(label)[0]+".csv", os.path.splitext(newpath)[0]+".csv")
            
            elif aug =='rescaled':
                # downscaled_
                apx = "rescaled_"
                newname = apx+label.split(dataset_path)[-1].strip(r"\\")
                newpath = os.path.join(imgaug_dataset_path, newname)
                if not os.path.exists(os.path.dirname(newpath)):
                    os.makedirs(os.path.dirname(newpath))
                ## change file path in labels
                df = pd.read_hdf(label)
                df.index = df.index.set_levels(apx + df.index.levels[1], level=1)
                df.to_hdf(newpath, key = "df_with_missing", mode="w")
                ## update rescaled label coords
                rescaledlcframelabels(rescale_factor, [newpath], output=False)
                ## save csv
                pd.read_hdf(newpath).to_csv(os.path.splitext(newpath)[0]+".csv")

            else:
                pass
        print(f'Label set successfully generated for {aug}!')

    return

def createlabeledvideo():
    '''
    TODO
    '''
    import matplotlib.pyplot as plt

    data = {'X': [200, 246, 387, 86, 100], 'Y': [100, 200, 34, 98, 234], 'Texture': [0,1,2,3,4]}
    img = plt.imread("img.jpg")
    # Define symbols and colors as you want.
    # Item at ith index corresponds to that label in 'Texture'.
    color = ['y', 'r', 'b', 'g', 'm']
    marker = ['o', 'v', '1', 's', 'p']
    # Zip the data, returns a generator of paired (x_i, y_i, texture_i)
    data_zipped = zip(*(data[col] for col in data))
    #
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 400, 0, 300])
    # 
    for x,y, texture in data_zipped:
        ax.plot(x, y, marker[texture],color=color[texture])
    plt.show()
    return

def createrescaledvideosamples(videos, scales):
    '''
    Creates new rescaled video samples with scale in format `960:540` or `960:-1`.
    Useful to test DLC performance on different video resolutions.
    '''
    for video in videos:
        for scale in scales:
            output = os.path.join(os.path.dirname(video), 'resacled_'+scale.replace(':', '_')+'_'+os.path.basename(video))
            command = f"ffmpeg -i {video} -vf scale={scale} {output}"
            # NOTE that this is a lossy procedure with re-encoding, specify crf to reduce compression loss
            os.system(command)
    return
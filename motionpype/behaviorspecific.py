"""
MotionPype
© Guillermo Hidalgo-Gadea, Department of Biopsychology
Institute of Cognitive Neuroscience, Ruhr University Bochum

source: https://gitlab.ruhr-uni-bochum.de/ikn/motionpype.git
Licensed under GNU Lesser General Public License v2.1
"""

# list of all required libraries, reduce if possible 
import os
import scipy.signal
import warnings
import numpy as np
import pandas as pd
#import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint


def fit_quadrants(step, grain, points):
    '''
    fits a quadrant grid in plane between points
    TODO find alternative binning
    '''
    # create points with x and y coordinates
    pointmap = []
    shape = Polygon(points)
    
    # create a horizontal, equidistant line of point between with `grain` number of points 
    for x in np.linspace(min(MultiPoint(points).bounds), max(MultiPoint(points).bounds), grain, dtype = float):

        # create a vertical, equidistant line of point between with `grain` number of points 
        for y in np.linspace(min(MultiPoint(points).bounds), max(MultiPoint(points).bounds), grain, dtype = float):
            
            # within the double loop, define point with x and y coordinates
            p = Point(x,y)

            # filter points outside the original hexagon arena
            if shape.contains(p):
                pointmap.append(p)

    # initialize quad_size
    quad_size = 0
    
    # initialize quadrants
    quadrants = [p.buffer(quad_size, cap_style=3) for p in pointmap]

    while MultiPolygon(quadrants).is_valid:
        quad_size += step

        # create quadrant grid
        quadrants = [p.buffer(quad_size, cap_style=3) for p in pointmap]

    return pointmap, quad_size, quadrants

'''
EXAMPLE
pointmap, quad_size, quadrants = fit_quadrants(step, grain, points)
'''

def extractpointcoordinates(df, points):
    coords = []
    for ref in points:
        x = df.loc[0,df.columns.str.contains(ref+'_x')]
        y = df.loc[0,df.columns.str.contains(ref+'_y')]
        z = df.loc[0,df.columns.str.contains(ref+'_z')]
        coords.append(np.array(x, y, z))

    return coords
'''
EXEMPLE
points = ['cA','cB','cC','cD','cE','cF']
coords = extractpointcoordinates(df, points)
'''

def getactivitythresholds():
    '''
    Returns activity thresholds from centroid
    TODO
    see here: http://emilygraceripka.com/blog/16
    '''
    # calculate distribution of centroid

    # fit distributions

    # get intersections and outliers

    return


def speedsegmentation(centroid, fps, rest_thr, outlier_thr, plot = True):
    '''
    Returns  
    '''
    # calculate speed as displacement between subsequent frames
    x = centroid[centroid.columns[centroid.columns.str.contains("x")]].values
    y = centroid[centroid.columns[centroid.columns.str.contains("y")]].values
    z = centroid[centroid.columns[centroid.columns.str.contains("z")]].values
    # conversion Hz to sec
    u = np.diff(x, n=1, axis=0) * fps 
    v = np.diff(y, n=1, axis=0) * fps
    w = np.diff(z, n=1, axis=0) * fps
    
    # calculate absolute speed
    total_speed = np.sqrt(u**2 + v**2 + w**2)
    # get rest
    rest = np.copy(total_speed)
    rest[rest>rest_thr] = np.nan
    # get movement
    movement = np.copy(total_speed)
    movement[movement<rest_thr] = np.nan
    movement[movement>outlier_thr] = np.nan

    # TODO calculate stop events and bouts

    if plot: 
        # plot Kinematics
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20,20))
        # plot 1 Velocity
        ax1.set(title = 'Velocity in x-, y-, and z-coordinates', xlabel=f'frames [{fps}Hz]', ylabel='velocity [mm/sec]');
        ax1.plot(u, label = 'x-coord')
        ax1.plot(v, label = 'y-coord')
        ax1.plot(w, label = 'z-coord')
        ax1.legend();

        # plot 2 Speed
        ax2.set(title = 'Absolute speed 3D space', xlabel=f'frames [{fps}Hz]', ylabel='speed [mm/sec]');
        ax2.plot(total_speed, label = 'speed')
        ax2.axhline(y=outlier_thr, color= 'red', label = 'outlier threshold')
        ax2.legend()

        # plot 3 Speed distribution
        ax3.set(title = 'Speed distribution', xlabel='speed [mm/sec]', ylabel='abs. frequency');
        ax3.hist(total_speed, bins = 1000, label = 'speed');
        ax3.axvline(x=rest_thr, color= 'black', label = 'inactivity threshold')
        ax3.axvline(x=outlier_thr, color= 'red', label = 'outlier threshold')
        #ax3.set_ylim(0,500)
        #ax3.set_xlim(0, 2000)
        ax3.legend()

        # plot 4 Segmentation
        ax4.set(title = 'Absolute speed segmentation', xlabel=f'frames [{fps}Hz]', ylabel='speed [mm/sec]');
        ax4.plot(movement, label = 'movement')
        ax4.plot(rest, label = 'inactivity')
        ax4.axhline(y=rest_thr, color= 'black', label = 'inactivity threshold')
        ax4.axhline(y=outlier_thr, color= 'red', label = 'outlier threshold')
        ax4.legend();

        fig.tight_layout()
        plt.show()
    return total_speed, rest, movement

'''
EXAMPLE
total_speed, rest, movement = restsegmentation(centroid, fps, rest_thr, outlier_thr, plot = True)
'''

def rescalefps(df, original_fps, new_fps):
    '''
    Returns a subset of the dataset matching new framerate
    '''
    samples = len(df)
    binning = original_fps/new_fps
    rescale_samples = int(samples/binning)
    rescale_idx = np.linspace(0, samples-1, num = rescale_samples)
    rescaled = df.iloc[rescale_idx].reset_index(drop=True)
    
    return rescaled

'''
EXAMPLE
new_df = rescalefps(df, original_fps, new_fps)
'''

def spatialdistribution(centroid, bouts):
    '''
    Return spatial distribution during bout
    TODO
    '''
    # get coordinates in every time frame
    for t in range(len(res_pigeon_bouts1)):
        p = Point(res_pigeon_bouts1.iloc[t,:])
        # assign quadrant that contains pigeon location
        boollist = quadgrid.contains(p)
        [visits1.append(i) for i, x in enumerate(boollist) if x]

    # count visit per quadrant index
    counts = pd.Series(visits1).groupby(visits1).size().reindex(quadgrid.index, fill_value=0)

    # create GeoPandas Data Frame
    boutcount1 = gpd.GeoDataFrame(pd.concat([counts, quadgrid], axis=1))
    boutcount1.columns = ["visits", "geometry"]

    return

def trimdatafortimesync():
    '''
    Creates trimmed dataframes given a list of start frames
    Useful to time-synch video with external clock
    TODO
    '''
    timesync = {
                '2022_02_02_P087_camL': 303,
                '2022_02_02_P087_camR': 303,
                '2022_02_02_P098_camL': 184,
                '2022_02_02_P098_camR': 183,}
    # load parameters from config
    configtoml = toml.load(configfile)
    pose_dir = configtoml['pipeline']['pose_2d_merged']
    output_dir = configtoml['pipeline']['pose_2d_sync']

    # trim data
    # loop over session directories in project
    for session in os.listdir(projectpath):
        try:
            print(f'Processing session: {session}')
            datadir = os.path.join(projectpath, session, pose_dir)
            posedata = os.listdir(datadir)
            outputdir = os.path.join(projectpath, session, output_dir)

            # create output dir
            if os.path.isdir(outputdir):
                pass
            else:
                os.makedirs(outputdir)

            for file in posedata:
                datafile = os.path.join(projectpath, session, pose_dir, file)

                # read data
                df = pd.read_hdf(datafile)

                # get match
                cam = file.split('_')[-1].split('.')[0]
                match = session + '_' + cam

                # get startframe
                startframe = timesync[match]

                # trim df
                new_df = df[startframe:-1].reset_index(drop=True)

                # save new df
                outputfile = outputdir + '/' + 'sync_' + file
                new_df.to_hdf(outputfile, key ='df', mode='w')
                print(f'File saved: {outputfile}')
            
        except:
            print('oops')
    
    return

def read_anipose_data(file, reference_points = ['']):
    '''
    Returns relevant subsets of anipose csv file and separates tracking in pose and reference when `reference_points` provided ()
    '''
    # read data
    df = pd.read_csv(file)
    # subset triangulation parameters
    center = df.loc[:,df.columns.str.contains('center')]
    M = df.loc[:,df.columns.str.contains('M_')] # rotation matrix after triangulation
    error = df.loc[:,df.columns.str.contains('error')]
    ncams = df.loc[:,df.columns.str.contains('ncams')]
    score = ncams = df.loc[:,df.columns.str.contains('score')]
    
    # subset pose and reference
    scene = df.loc[:,~df.columns.str.contains('score|error|ncams|fnum|center|M_')]
    if len(reference_points[len(reference_points)-1]) < 1:
        reference = pose = scene
    else:
        reference = scene.loc[:,scene.columns.str.contains(('|').join(reference_points))]
        pose = scene.loc[:,~scene.columns.str.contains(('|').join(reference_points))]

    return error, ncams, score, reference, pose

def plane_transform(s, p1, p2, data):
    '''
    Returns transformed data and rotation matrix aligned to new plane
    given by s, p1 and p2   
    '''
    # calculate vectors
    v = p1 - s
    u = p2 - s
    n = np.cross(v, u)
    w = np.cross(v,n)

    # set unit length
    v = v/np.linalg.norm(v)
    u = u/np.linalg.norm(u)
    w = w/np.linalg.norm(w)
    n = n/np.linalg.norm(n)

    # create tranformation matrix
    M = np.transpose([v, w, n]).reshape((3,3))

    # tranform data
    transformed = data@M

    # reset coordinate system
    transformed = transformed - transformed[0:1][2][0]
    transformed = pd.DataFrame({ 'x': transformed[0], 'y': transformed[1],  'z': transformed[2]})
    
    return M, transformed

def kinematics(df, fps, inactivity_thresh = 1):
    '''
    Return kinematics from df
    TODO fix framerate conversion
    '''
    # calculate speed in mm/frame
    vx = np.diff(df.x, n=1, axis=0)
    vy = np.diff(df.y, n=1, axis=0)
    vz = np.diff(df.z, n=1, axis=0)
    speed = np.sqrt(vx**2 + vy**2 + vz**2) 

    # smooth kinematics
    k_smooth = int(fps/4)
    if k_smooth % 2 == 0:
        k_smooth += 1
    speed = scipy.signal.medfilt(speed, k_smooth)

    # calculate inactivity index
    inactivity = speed.copy().astype(bool)
    inactivity = speed.copy()
    inactivity[speed<inactivity_thresh] = False
    inactivity[speed>=inactivity_thresh] = True

    # calculate traveled path
    path = sum(speed) / 1000 # in m

    # calculate acceleration in mm/frame**2
    ax = np.diff(vx, n=1, axis=0)
    ay = np.diff(vy, n=1, axis=0)
    az = np.diff(vz, n=1, axis=0)
    acceleration = np.sqrt(ax**2 + ay**2 + az**2) 

    return path, speed, acceleration, inactivity

def exponential_smoothing(data, alpha):
    '''
    exponential smoothing with yt = alpha * xt + (1-alpha) * yt-1
    Smaller alpha values give more weight to historical observations and result in smoother trajectories
    '''
    data = np.array(data)
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

def exponential_smoothing_2d(data, alpha):
    data = np.array(data)
    smoothed = np.zeros_like(data)
    smoothed[0, :] = data[0, :] 
    
    for i in range(1, data.shape[0]):
        smoothed[i, :] = alpha * data[i, :] + (1 - alpha) * smoothed[i-1, :]
    
    return smoothed

def subsetlikelitracking(df, pcutoff, output = False):
    '''
    Returns a subset of coordinates with high tracking likelihood
    '''
    likes = [col for col in df.columns if 'likelihood' in col]
    df_likes = df[likes]
    likes = df_likes.median(axis = 0)
    idx = likes[likes >= pcutoff].index.get_level_values(1)
    subset = [col for col in df.columns for i in idx if i in col]
    subset_df = df[subset].copy()
    if output:
        print(f'{(len(subset) / len(df.columns)*100):2f} % of bodyparts above pcutoff: {pcutoff}')
    
    return subset_df

def reduceposetocentroid(df, subset = [], med_smooth=0):
    '''
    Returns the spatial median of a subset of points
    '''
    # subset data by keypoints
    if len(subset) > 0:
        data = df.loc[:,df.columns.str.contains(('|').join(subset))]
    else:
        data = df

    xcols = [col for col in data.columns if 'x' in col]
    ycols = [col for col in data.columns if 'y' in col]
    zcols = [col for col in data.columns if 'z' in col]

    # apply spatial median
    x = np.nanmedian(data[xcols], axis = 1)
    y = np.nanmedian(data[ycols], axis = 1)
    z = np.nanmedian(data[zcols], axis = 1)

    # smoothen the cenrtroid
    if med_smooth > 0:
        x = scipy.signal.medfilt(x, kernel_size = med_smooth)
        y = scipy.signal.medfilt(y, kernel_size = med_smooth)
        z = scipy.signal.medfilt(z, kernel_size = med_smooth)
    else:
        pass
    centroid = pd.DataFrame({'x':x, 'y':y, 'z':z})

    return centroid

def compoundfiltering(df, bodypart = 'H(Head)', spatial_range = 0.99, temporal_range = 0.99, pcutoff = 0.10, likelihood_range = 0.10, output = False):
    '''
    Return filtered trajectory
    '''
    warnings.filterwarnings("ignore", "Mean of empty slice")

    # A: It is weird for a bodypart to move too far away
    spat_scale = [  round(0+((1-spatial_range)*100)/2, 2),
                    round(100-((1-spatial_range)*100)/2, 2)]
    # B: It is weird for a bodypart to move too fast
    temp_scale = [  round(0+((1-temporal_range)*100)/2, 2),
                    round(100-((1-temporal_range)*100)/2, 2)]
    # C: Likelihood
    likelihood_range = likelihood_range *100

    # subset keypoint
    kp = [col for col in df.columns if bodypart in col]
    keypoint = df[kp].droplevel([0,1], axis =1)
    keypoint

    # calculate smooth centroid from pcutoff
    goodtracking = subsetlikelitracking(df, pcutoff)
    centroid = reduceposetocentroid(goodtracking, med_smooth = 7)

    # find distribution percentiles and use as offset_threshold
    space_offset = (centroid-keypoint).values.reshape(-1)
    space_offset_th = np.nanpercentile(space_offset, spat_scale).round(2)

    d1s = np.diff(keypoint, n=1, axis = 0)
    time_offset = np.insert(d1s.reshape(-1), list(range(0,d1s.shape[1])), np.nanmean(d1s))
    time_offset_th = np.nanpercentile(time_offset, temp_scale).round(2)

    like_offset_th = np.nanpercentile(keypoint.likelihood, likelihood_range).round(2)

    if output:

        # plot results
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,5))

        ax1.hist(space_offset, bins = 100);
        ax1.axvline(x = space_offset_th[0], c = 'red', lw = 0.4, label = f'offsets: {space_offset_th}')
        ax1.axvline(x = space_offset_th[1], c = 'red', lw = 0.4, label = f'{spat_scale[1]-spat_scale[0]}% CI ')
        ax1.set(xlabel = 'Spatial offset in px')
        ax1.legend()

        ax2.hist(time_offset, bins = 100);
        ax2.axvline(x = time_offset_th[0], c = 'red', lw = 0.4, label = f'offsets: {time_offset_th}')
        ax2.axvline(x = time_offset_th[1], c = 'red', lw = 0.4, label = f'{temp_scale[1]-temp_scale[0]}% CI ')
        ax2.set(xlabel = 'Temporal offset in px/frame')
        ax2.legend()

        ax3.hist(keypoint.likelihood, bins = 100);
        ax3.axvline(x = like_offset_th, c = 'red', lw = 0.4, label = f'offset: {like_offset_th}, \n lower {likelihood_range}%')
        ax3.set(xlabel = 'Tracking likelihood')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    else:
        pass

    # filter data
    offset = keypoint.loc[:,'x':'y']-centroid
    spatial_outlier_idx = np.isfinite(offset[offset<space_offset_th[0]]) +  np.isfinite(offset[offset>space_offset_th[1]])

    d1s = pd.DataFrame(np.pad(np.diff(keypoint, n=1, axis = 0), ((1,0),(0,0)), 'mean'), columns = ['x', 'y', 'likelihood'])
    temp_outlier_idx = np.isfinite(d1s[d1s<time_offset_th[0]]) + np.isfinite(d1s[d1s>time_offset_th[1]])

    likelihood_outlier_idx = keypoint < like_offset_th
    likelihood_outlier_idx.x = likelihood_outlier_idx.y = likelihood_outlier_idx.likelihood

    comb_outlier_idx = np.invert(np.isfinite(keypoint[temp_outlier_idx])) & np.invert(np.isfinite(keypoint[spatial_outlier_idx])) & np.invert(np.isfinite(keypoint[likelihood_outlier_idx]))
    compound_filter = keypoint.loc[:,'x':'y'][comb_outlier_idx]

    interpolated = compound_filter.interpolate(method='spline', order = 1, limit_direction = 'both')

    if output:
        # plot filtered traces
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, figsize = (20,20))

        # spatial filtering
        ax1.plot(keypoint.loc[:,'x':'y'], alpha = 0.7)
        ax1.plot(centroid, c = 'gray', alpha = 0.5)
        ax1.plot(keypoint[spatial_outlier_idx], c = 'red')
        ax1.set(title = 'Spatial offset')

        # temporal filtering
        ax2.plot(keypoint.loc[:,'x':'y'], alpha = 0.7)
        ax2.plot(centroid, c = 'gray', alpha = 0.5)
        ax2.plot(keypoint.loc[:,'x':'y'][temp_outlier_idx], c = 'red')
        ax2.set(title = 'Temporal offset')

        # likelihood filtering
        ax3.plot(keypoint.loc[:,'x':'y'], alpha = 0.7)
        ax3.plot(centroid, c = 'gray', alpha = 0.5)
        ax3.plot(keypoint.loc[:,'x':'y'][likelihood_outlier_idx], alpha = 0.7, c ='red')
        ax3.set(title ='Likelihood offset')

        # subset 
        ax4.plot(compound_filter , alpha = 0.7)
        ax4.plot(centroid, c = 'gray', alpha = 0.5)
        ax4.set(title = 'Compound Filter')

        # interpolate
        ax5.plot(interpolated, alpha = 0.7, c = 'red')
        ax5.plot(compound_filter, c ='black')
        ax5.plot(centroid, c = 'gray', alpha = 0.5)

        plt.show()
        plt.tight_layout()
    else:
        pass

    return interpolated, compound_filter, keypoint

def comparefilters(interpolated, filtered, coordinate = 'x', n = 5, output = True):
    trace = interpolated[coordinate]
    filtered = filtered[coordinate]

    # exponential filter range
    alpha_range = np.linspace(0.9,0.1,n)
    # median filter range
    kernel = np.arange(3, n*2, 2)

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (20,20))

    # exponential filter
    ax1.set(title = 'Exponential smoothing residuals')
    offset = 0
    for alpha in alpha_range:
        exp_filt = exponential_smoothing(trace, alpha=alpha)
        diff = trace-exp_filt
        offset += (np.max(diff) - np.min(diff))*0.7
        ax1.plot(diff + offset, alpha = 0.7, label = f'alpha = {alpha:.2f}')
        
    ax1.legend(loc='lower center',ncol=3,)
    ax1.yaxis.set_ticklabels([])

    # reference
    ax2.set(title = 'Filtered tracking')
    ax2.plot(trace, color = 'red', label = 'interpolation', alpha = 0.7)
    ax2.plot(filtered , color = 'black', alpha = 0.8, lw = 2,label = 'filtered tracking')
    ax2.legend();

    # median filter
    ax3.set(title = 'Median smoothing residuals')
    offset = 0
    for k in kernel:
        med_filt = scipy.signal.medfilt(trace, kernel_size=k)
        diff = (trace-med_filt)
        offset += (np.max(diff) - np.min(diff))*0.7
        ax3.plot(diff - offset , alpha = 0.7, label = f'k = {k}')
    ax3.legend(loc='lower center', ncol=3)
    ax3.yaxis.set_ticklabels([])

    plt.show()
    plt.tight_layout()
    return

def comparefiltersmoothing(interpolated, filtered, keypoint, coordinate = 'x', k = 7, a = 0.6, CI = 90, y_offset = 100, output = False):
    trace = interpolated[coordinate]
    filtered = filtered[coordinate]
    raw = keypoint[coordinate]

    smooth_scale = [(100-CI)/2, 100-(100-CI)/2]

    # exponential smoothing
    exp_filt = pd.Series(exponential_smoothing(trace, alpha=a))
    exp_diff = trace-exp_filt
    smooth_offset_th = np.nanpercentile(exp_diff, smooth_scale).round(2)
    exp_smooth = trace.copy()
    exp_smooth[exp_diff < smooth_offset_th[0]] = exp_filt[exp_diff < smooth_offset_th[0]]
    exp_smooth[exp_diff > smooth_offset_th[1]] = exp_filt[exp_diff > smooth_offset_th[1]]

    # median smoothing
    med_filt = pd.Series(scipy.signal.medfilt(trace, kernel_size=k))
    med_diff = trace-med_filt
    smooth_offset_th = np.nanpercentile(med_diff, smooth_scale).round(2)
    med_smooth = trace.copy()
    med_smooth[med_diff <smooth_offset_th[0]] = med_filt[med_diff<smooth_offset_th[0]]
    med_smooth[med_diff >smooth_offset_th[1]] = med_filt[med_diff>smooth_offset_th[1]]

    if output:
        # plot results
        fig, (ax1, ax2) = plt.subplots(2,1, figsize = (20,10))
        ax1.set(title = 'Smoothing and filter comparison')
        ax1.plot(raw + y_offset , c = 'gray', alpha = .6, label = 'raw tracking')
        ax1.plot(trace, c = 'red')
        ax1.plot(filtered, c = 'black', label = 'filtered tracking')
        ax1.plot(exp_filt - y_offset, alpha = .6, label = 'exp_filtered')
        ax1.plot(exp_smooth- 2*y_offset, alpha = .6, label = 'exp_smoothed')
        ax1.plot(med_filt-3*y_offset, alpha = .6, label = 'med_filtered')
        ax1.plot(exp_smooth- 4*y_offset, alpha = .6, label = 'med_smoothed')
        ax1.legend();

        ax2.set(title = 'Smoothing and filter residuals')
        offset = 0
        offset+=(np.max(trace-exp_filt) - np.min(trace-exp_filt))
        ax2.plot(trace-exp_filt - offset, alpha = .6, label = 'exp_filtered')
        offset+=(np.max(trace-exp_smooth) - np.min(trace-exp_smooth))
        ax2.plot(trace-exp_smooth - offset, alpha = .6, label = 'exp_smoothed')
        offset+=(np.max(trace-med_filt) - np.min(trace-med_filt))
        ax2.plot(trace-med_filt - offset, alpha = .6, label = 'med_filtered')
        offset+=(np.max(trace-med_smooth) - np.min(trace-med_smooth))
        ax2.plot(trace-med_smooth - offset, alpha = .6, label = 'med_smoothed')
        ax2.yaxis.set_ticklabels([])
        ax2.legend()
    else:
        pass
    
    return exp_filt, exp_smooth, med_filt, med_smooth

def comparespatialfilter(interpolated, compound_filter, keypoint, alpha=0.7, smooth_scale = [5, 95]):
    # compare 2d traces from 1D interpolations
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize = (10,10))

    ax1.scatter(compound_filter.x, compound_filter.y, color = 'black')
    ax1.plot(compound_filter.x, compound_filter.y, linestyle='-', lw = 3, color = 'black')
    ax1.plot(keypoint.x, keypoint.y, linestyle='-', lw = 0.6,color = 'gray')
    ax1.set(title = 'Spatiotemporal filter')

    ax2.scatter(compound_filter.x, compound_filter.y, color = 'black')
    ax2.plot(interpolated.x, interpolated.y, linestyle='-', color = 'red')
    ax2.set(title = 'Spline interpolation')

    x = exponential_smoothing(interpolated.x, alpha=alpha)
    y = exponential_smoothing(interpolated.y, alpha=alpha)
    ax3.plot(interpolated.x, interpolated.y, linestyle='-', color = 'red')
    ax3.plot(x, y, linestyle='-', color = 'darkgreen')
    ax3.set(title = 'Exponential filter over interpolation')

    x_smooth = interpolated.x.copy()
    y_smooth = interpolated.y.copy()
    exp_diff = pd.DataFrame(x_smooth-x).join(y_smooth-y)
    smooth_offset_th = np.nanpercentile(exp_diff, smooth_scale).round(2)
    x_smooth[exp_diff.x < smooth_offset_th[0]] = x[exp_diff.x < smooth_offset_th[0]]
    y_smooth[exp_diff.y > smooth_offset_th[1]] = y[exp_diff.y > smooth_offset_th[1]]

    ax4.plot(x_smooth, y_smooth, linestyle='-', color = 'orange')
    ax4.plot(interpolated.x, interpolated.y, linestyle='-', lw =0.6, color = 'red')
    ax4.plot(x, y, linestyle='-', lw =0.6, color = 'darkgreen')
    ax4.scatter(compound_filter.x, compound_filter.y, color = 'black')
    ax4.set(title = 'Exponential smoothing')
    plt.tight_layout()
    return

def applyfilterstohdf(file, coords = ['x', 'y'], spatial_range = 0.99, temporal_range = 0.99, pcutoff = 0.7, likelihood_range = .2, k = 9, a = .7, CI = 50):
    '''
    Returns a saved h5 file in DLC format after applying filters from compundfiltering and filtersmoothing to entire df
    '''
    # read df
    df = pd.read_hdf(file)
    scorer = df.columns.levels[0][0]
    filtered_df = df.copy()

    # run filter for all bodyparts
    for bp in df.columns.levels[1]:
        interpolated, filtered, keypoint = compoundfiltering(df, bp, spatial_range, temporal_range, pcutoff, likelihood_range)
        for coord in coords:
            exp_filt, exp_smooth, med_filt, med_smooth = comparefiltersmoothing(interpolated, filtered, keypoint, coord, k, a, CI)
            filtered_df[(scorer, bp, coord)] = exp_smooth

    # save as filtered df
    outfile = os.path.splitext(file)[0] + '_filtered.h5'
    filtered_df.to_hdf(outfile, key = "df_with_missing", mode="w")
    return

def headvector(head_data):
    '''
    head_data: tuple of numpy arrays 
    headcenter: numpy array with x,y,z coordinates of the center
    headdirection: numpy array with x,y,z coordinates of head direction vector 
    TODO
    '''
    x_head_data, y_head_data, z_head_data = head_data
    headcenter = []
    headdirection = []
    for i in range(np.shape(x_head_data)[1]):
        data = np.stack((x_head_data[i], y_head_data[i], z_head_data[i]), axis=1)
        # calculate head center
        center = data.mean(axis=0)

        # check for alignement HARDCODED
        neck = (data[0] - center)/np.linalg.norm((data[0] - center))
        beak = (data[4] - center)/np.linalg.norm((data[4] - center))

        # calculate singular value decomposition (SVD)
        u, s, vh = np.linalg.svd(data - center)

        # check head direction goes from neck to beak
        if np.dot(vh[0],beak) > np.dot(vh[0],neck):
            correction_factor = 1
        else:
            correction_factor = -1

        # aggregate data over loop
        headdirection.append(correction_factor*vh[0])
        headcenter.append(center)
    
    return np.array(headcenter), np.array(headdirection)

def normalize_decompose_vector(vector, origin):
    '''
    TODO
    '''
    norm_vector = []
    azimuth = []
    elevation = []

    for i, vec in enumerate(vector):
        # align vector to origin
        centered_vec = vec - origin[i]
        # normalize
        norm_vec = centered_vec/np.linalg.norm(centered_vec)
        norm_vector.append(norm_vec)

        # xy-plane decomposition
        #rad_azim = np.arccos(np.dot(v, w)/(np.linalg.norm(v)*np.linalg.norm(w))) # note: this angle is only betwen 0 and 180,  When the angle between the vectors is greater than 180 degrees, the cross product flips over to point in the opposite direction.
        #rad_azim = (np.arctan2(v[1],v[0]) - np.arctan2(w[1],w[0]))
        rad_azim = np.arctan2(centered_vec[1],centered_vec[0])
        azimuth.append(rad_azim)

        # yz plane decomposition    
        #rad_elev = np.arccos(np.dot(v, w)/(np.linalg.norm(v)*np.linalg.norm(w))) # note: this angle is only betwen 0 and 180
        #rad_elev = (np.arctan2(v[1],v[0]) - np.arctan2(w[2],w[1]))
        rad_elev = np.arctan2(centered_vec[2],np.sqrt(centered_vec[1]**2+centered_vec[0]**2))

        elevation.append(rad_elev)

    return np.array(norm_vector), np.array(azimuth), np.array(elevation)

def get_svd_axes(df, maxlength = 30000, axlength = 500):
    '''
    Returns the new x, y, z axes from the SVD of the df given
    TODO: check directions
    '''
    # calculate center of pointcloud
    center = np.array(df.mean(axis=0))
    
    # downscale if needed
    n = int(np.ceil(len(df)/maxlength))
    downscaled = df.iloc[::n, :].reset_index(drop=True)
    
    # calculate singular value decomposition (SVD)
    u, s, vh = np.linalg.svd(downscaled - center)

    return vh*axlength

def findpointin3DROI(points, ROI, smooth=1):
    '''
    Return vector of point in ROI when point is within 3dROI
    ROI needs to be defined with ROI[0] as origin and edges pointing out
    Scalar or dot product calculates the angle between each point and its axis, 
    and the angle between the centroid and each of these axis must lie between the limits. 
    '''
    roilist = []
    ax1 = ROI[1]-ROI[0]
    ax2 = ROI[2]-ROI[0]
    ax3 = ROI[3]-ROI[0]

    # calculate inner products for each eadge limits
    ll_ax1 = ROI[0]@ax1
    ul_ax1 = ROI[1]@ax1
    ll_ax2 = ROI[0]@ax2
    ul_ax2 = ROI[2]@ax2
    ll_ax3 = ROI[0]@ax3
    ul_ax3 = ROI[3]@ax3

    # find if inner product of p falls within range for every dimension
    for n in range(len(points)):
        p = np.array(points.iloc[n,:])
        if ll_ax1 <= p@ax1 <= ul_ax1:
            if ll_ax2 <= p@ax2 <= ul_ax2:
                if ll_ax3 <= p@ax3 <= ul_ax3:
                    x = True
                else:
                    x = False
            else:
                x = False
        else:
            x = False

        roilist.append(x)

    # smooth to avoid jumps
    int_roilist = [int(i) for i in roilist]
    filt_int_roilist = scipy.signal.medfilt(int_roilist, kernel_size=smooth)
    filt_roilist = [bool(i) for i in filt_int_roilist]
    
    return filt_roilist

def findpointinline(p1, p2, d):
    '''
    Returns a point p at a distance d from p1 on a line between p1 and p2 
    '''
    r = d/np.linalg.norm(p2 - p1)
    p = p1 + r * (p2-p1)

    return p

def findplatformROI(p1, p2, width, height, depth, padding, elevation):
    '''
    Returns the ROI corners for a 3D rectangle centered between p1 and p2
    '''
    width += padding
    depth += padding
    height += padding

    # width edge aligned on p1-p2, with p1 closer to reference
    roi_corner_0 = findpointinline(p1, p2, (np.linalg.norm(p1 - p2) - width )/2) + np.array([0, 0, elevation - height/3]) # reference
    roi_corner_1 = findpointinline(p1, p2, (np.linalg.norm(p1 - p2) + width )/2) + np.array([0, 0, elevation - height/3]) # width

    # height above reference
    roi_corner_2 = roi_corner_0 + np.array([0, 0, height]) # height

    # depth orthogonal to height and width
    ax = np.cross(roi_corner_0-roi_corner_1, roi_corner_0-roi_corner_2)
    norm = ax / np.linalg.norm(ax)
    roi_corner_3 = roi_corner_0 + (depth * norm) # depth
    
    # order corner 0 as reference for function findpointin3DROI(points, ROI)
    roi = np.array([roi_corner_0, roi_corner_1, roi_corner_2, roi_corner_3]).astype('int32')
    
    return roi


def draw3DROI(ROI):
    '''
    Returns all corners and edges to draw the ROI
    '''
    
    return

def drawreference(reference, top):
    '''
    '''
    x = np.array(reference.loc[0,reference.columns.str.contains("_x")])
    y = np.array(reference.loc[0,reference.columns.str.contains("_y")])
    z = np.array(reference.loc[0,reference.columns.str.contains("_z")])

    corners = (np.append(x,x), np.append(y, y), np.append(z, z+top))

    # connect edges
    floor = (np.append(x,x[0]), np.append(y, y[0]), np.append(z, z[0]))
    ceiling = (np.append(x,x[0]), np.append(y, y[0]), np.append(z+top, z[0]+top))
    walls = np.array([[floor[0][0], ceiling[0][0], floor[0][1], ceiling[0][1], floor[0][2], ceiling[0][2], 
                floor[0][3], ceiling[0][3], floor[0][4], ceiling[0][4], floor[0][5], ceiling[0][5], 
                floor[0][6], ceiling[0][6], floor[0][5], ceiling[0][5], floor[0][4], ceiling[0][4],
                floor[0][3], ceiling[0][3], floor[0][2], ceiling[0][2], floor[0][1], ceiling[0][1], floor[0][0]], 
                [floor[1][0], ceiling[1][0], floor[1][1], ceiling[1][1], floor[1][2], ceiling[1][2], 
                floor[1][3], ceiling[1][3], floor[1][4], ceiling[1][4], floor[1][5], ceiling[1][5], 
                floor[1][6], ceiling[1][6], floor[1][5], ceiling[1][5], floor[1][4], ceiling[1][4],
                floor[1][3], ceiling[1][3], floor[1][2], ceiling[1][2], floor[1][1], ceiling[1][1], floor[1][0]],
                [floor[2][0], ceiling[2][0], floor[2][1], ceiling[2][1], floor[2][2], ceiling[2][2], 
                floor[2][3], ceiling[2][3], floor[2][4], ceiling[2][4], floor[2][5], ceiling[2][5], 
                floor[2][6], ceiling[2][6], floor[2][5], ceiling[2][5], floor[2][4], ceiling[2][4],
                floor[2][3], ceiling[2][3], floor[2][2], ceiling[2][2], floor[2][1], ceiling[2][1], floor[2][0]],])
    
    edges = np.concatenate((floor, ceiling, walls), axis= 1)
    
    return corners, edges
def drawroi(roi):
    '''
    Returns corners and edges of given ROI
    '''
    corners = np.array([roi[0],     #1
                        roi[1],    # 2
                        roi[1] + roi[3]-roi[0], #3
                        roi[3], # 4

                        roi[2], #5
                        roi[2] + roi[1]-roi[0],    # 6
                        roi[2] + roi[1]-roi[0] + roi[3]-roi[0],# 7
                        roi[2] + roi[3]-roi[0],# 8
                        ])

    edges = np.array([roi[0],     roi[1],
                        roi[1] + roi[3]-roi[0],
                        roi[3],     roi[0], 
                        roi[2], roi[2] + roi[1]-roi[0],
                        roi[2] + roi[1]-roi[0] + roi[3]-roi[0],
                        roi[2] + roi[3]-roi[0], roi[2],
                        roi[1],    roi[2] + roi[1]-roi[0],
                        roi[1] + roi[3]-roi[0],
                        roi[2] + roi[1]-roi[0] + roi[3]-roi[0],
                        roi[3],  roi[2] + roi[3]-roi[0],  
                        roi[0], roi[2] + roi[1]-roi[0], roi[1],
                        roi[2] + roi[1]-roi[0] + roi[3]-roi[0],
                        roi[1] + roi[3]-roi[0], 
                        roi[2] + roi[3]-roi[0], roi[3], roi[2],
                        roi[2] + roi[1]-roi[0] + roi[3]-roi[0],
                        roi[2] + roi[3]-roi[0],roi[2] + roi[1]-roi[0],
                        roi[1], roi[3], roi[0],roi[1] + roi[3]-roi[0],
                        ])
    return corners, edges

def plotscene(centroid, on_left_platform, on_right_platform, ref_corners, ref_edges, left_platform_edges, right_platform_edges, outputfile, PID, date, session, elevation= 20, azimuth=-30, save = False, output = True):

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elevation, azimuth)
    ax.scatter(centroid.x, centroid.y, centroid.z, color='black', s = 0.1)

    ax.scatter(centroid[on_left_platform].x, centroid[on_left_platform].y, centroid[on_left_platform].z, color = 'blue')
    ax.scatter(centroid[on_right_platform].x, centroid[on_right_platform].y, centroid[on_right_platform].z, color = 'red')

    # draw rois
    ax.plot(left_platform_edges[:,0], left_platform_edges[:,1], left_platform_edges[:,2], color = 'blue', alpha = 0.4);
    ax.plot(right_platform_edges[:,0], right_platform_edges[:,1], right_platform_edges[:,2], color = 'red', alpha = 0.4);

    # draw scene
    ax.scatter(ref_corners[0], ref_corners[1], ref_corners[2], color = 'tan')
    ax.plot(ref_edges[0], ref_edges[1], ref_edges[2], color = 'tan', lw = 1, linestyle = '--')
    ax.quiver(ref_corners[0][0], ref_corners[1][0], ref_corners[2][0], ref_corners[0][0] + 200, ref_corners[1][0], ref_corners[2][0])
    ax.quiver(ref_corners[0][0], ref_corners[1][0], ref_corners[2][0], ref_corners[0][0], ref_corners[1][0] + 200, ref_corners[2][0])
    ax.quiver(ref_corners[0][0], ref_corners[1][0], ref_corners[2][0], ref_corners[0][0], ref_corners[1][0], ref_corners[2][0]+ 200)
    ax.set(xlabel = 'x', ylabel = 'y', zlabel = 'z', title = f'Tracking for {PID}, {date} \ncondition: {session}');
    ax.grid(False)
    plt.tight_layout()
    
    if save:
        plt.savefig(outputfile, dpi = 1000, transparent = True, bbox_inches='tight')

    if not output:
        plt. close()

    return

def plotkinematics(centroid, acceleration, ref_corners, ref_edges, platform_A_edges, platform_B_edges, outputfile, PID, date, session, elevation= 20, azimuth=-30, save = False, output = True):

    # exponential filtering for smoother tracking
    acc = exponential_smoothing(abs(acceleration), alpha=0.7)
    # replace outliers
    outlier = np.nanpercentile(acc, 99.9).round(2)
    acc[acc>outlier] = np.median(acc)
    # mask lower quartiles
    lim = np.nanpercentile(acc, 25).round(2)
    acc[acc<lim] = np.median(acc)
    # normalize the rank
    rank = acc.argsort().argsort()
    rank_norm = rank/max(rank)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elevation, azimuth)
    p = ax.scatter(centroid.x, centroid.y, centroid.z, c = rank_norm, cmap='inferno', s = 0.5)

    # draw rois
    ax.plot(platform_A_edges[:,0], platform_A_edges[:,1], platform_A_edges[:,2], color = 'blue', lw = 0.5, alpha = 0.5);
    ax.plot(platform_B_edges[:,0], platform_B_edges[:,1], platform_B_edges[:,2], color = 'red', lw = 0.5, alpha = 0.5);

    # draw scene
    ax.scatter(ref_corners[0], ref_corners[1], ref_corners[2], color = 'tan')
    ax.plot(ref_edges[0], ref_edges[1], ref_edges[2], color = 'tan', lw = 1, linestyle = '--')
    ax.quiver(ref_corners[0][0], ref_corners[1][0], ref_corners[2][0], ref_corners[0][0] + 200, ref_corners[1][0], ref_corners[2][0])
    ax.quiver(ref_corners[0][0], ref_corners[1][0], ref_corners[2][0], ref_corners[0][0], ref_corners[1][0] + 200, ref_corners[2][0])
    ax.quiver(ref_corners[0][0], ref_corners[1][0], ref_corners[2][0], ref_corners[0][0], ref_corners[1][0], ref_corners[2][0]+ 200)
    ax.set(xlabel = 'x', ylabel = 'y', zlabel = 'z', title = f'Tracking for {PID}, {date} \ncondition: {session}');
    ax.grid(False)
    fig.colorbar(p, ax=ax, fraction=0.03, pad=0.15, label = 'acceleration by rank')
    plt.tight_layout()
    
    if save:
        plt.savefig(outputfile, dpi = 1000, transparent = True, bbox_inches='tight')

    if not output:
        plt. close()

    return

def plot_transitions(location_vec, outputfile, fps, PID, date, session, save = False, output = False):
    # platform transitions
    fig, ax = plt.subplots(1,1, figsize = (20, 5))
    ax.plot(location_vec['on_platform'], color = 'black')
    ax.plot(location_vec['platform_A'], color = 'blue', label = 'platform_A')
    ax.plot(location_vec['platform_B'], color = 'red', label = 'platform_B')
    [ax.axvline(x, color = 'blue', lw = 0.5) for x in list(location_vec['frame'][location_vec['visit_A']!=0])]
    [ax.axvline(x, color = 'red', lw = 0.5) for x in list(location_vec['frame'][location_vec['visit_B']!=0])]
    [ax.axvline(x, color = 'blue', lw = 0.5, linestyle= '--') for x in list(location_vec['frame'][location_vec['leave_A']!=0])]
    [ax.axvline(x, color = 'red', lw = 0.5, linestyle= '--') for x in list(location_vec['frame'][location_vec['leave_B']!=0])]
    ax.set(title = f'Transitions between Platforms for {PID}, {date}, condition: {session}', yticks = [0,1], ylabel = 'on platform', xlabel = 'time in frames')
    ax.set_xlim([0*60*fps, 25*60*fps])
    plt.legend();
    if save:
        plt.savefig(outputfile, dpi = 1000, transparent = True, bbox_inches='tight')

    if not output:
        plt. close()
    return

def calculateheadvectors(pose, subset = ['Head', 'UpperCere', 'LowerCere', 'BeakTip', 'LeftEye', 'RightEye', 'UpperNeck'], ksmooth = 1):
    '''
    TODO robustness in manual back to front vectors
    TODO vectorization
    '''
    
    # subset head data from pose
    if len(subset) > 0:
        data = pose.loc[:,pose.columns.str.contains(('|').join(subset))]
    else:
        data = pose

    xcols = [col for col in data.columns if '_x' in col]
    ycols = [col for col in data.columns if '_y' in col]
    zcols = [col for col in data.columns if '_z' in col]

    # smooth data
    if ksmooth > 1:
        x = scipy.signal.medfilt(data[xcols], kernel_size = ksmooth)
        y = scipy.signal.medfilt(data[ycols], kernel_size = ksmooth)
        z = scipy.signal.medfilt(data[zcols], kernel_size = ksmooth)
    else:
        x = data[xcols]
        y = data[ycols]
        z = data[zcols]
    
    # stacked numpay array of head data (row x body part x coord)
    headdata = np.stack((x, y, z), axis=2)

    # calculate head center as median (the point that minimizes the sum of distances to all the points in the cloud)
    headcenter_med = np.median(headdata, axis=1)

    # center data
    x_cent = x - headcenter_med[:,0][:,None]
    y_cent = y - headcenter_med[:,1][:,None]
    z_cent = z - headcenter_med[:,2][:,None]
    centered_data = np.stack((x_cent, y_cent, z_cent), axis=2)

    # calculate manual head vectors
    UpperNeck = np.array(data[data.columns[data.columns.str.contains('UpperNeck')]])
    Head = np.array(data[data.columns[data.columns.str.contains('Head')]])
    UpperCere = np.array(data[data.columns[data.columns.str.contains('UpperCere')]])
    LowerCere = np.array(data[data.columns[data.columns.str.contains('LowerCere')]])
    BeakTip = np.array(data[data.columns[data.columns.str.contains('BeakTip')]])

    # calculate head direction vectors from back to front
    v1 = Head - UpperNeck
    v2 = UpperCere - UpperNeck
    v3 = LowerCere - UpperNeck
    v4 = BeakTip - UpperNeck
    v5 = UpperCere - Head
    v6 = LowerCere - Head
    v7 = BeakTip - Head
    v8 = LowerCere - UpperCere
    v9 = BeakTip - UpperCere
    v10 = BeakTip - LowerCere

    # calculate mean vector from v1 to v10
    headdirection_mean = np.mean(np.stack((v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)), axis=0)
    headdirection_mean = headdirection_mean/np.linalg.norm(headdirection_mean)
    
    # TODO vectorization works, but check for axis on every frame anyways...

    headvectors = []
    for i in range(len(centered_data)):
        # calculate singular value decomposition (SVD)
        u, s, vh = np.linalg.svd(centered_data[i])
    
        # find vector in vh with largest dot product with headdirection_mean
        if 'headdirection_mean' in locals():
            if abs(np.dot(vh[0],headdirection_mean[i])) > abs(np.dot(vh[1],headdirection_mean[i])):
                if abs(np.dot(vh[0],headdirection_mean[i])) > abs(np.dot(vh[2],headdirection_mean[i])):
                    headvector = vh[0]
                else:
                    headvector = vh[2]
            elif abs(np.dot(vh[1],headdirection_mean[i])) > abs(np.dot(vh[2],headdirection_mean[i])):
                headvector = vh[1]
            else:
                headvector = vh[2]
        else:
            # best guess
            headvector = vh[0]

        # cehck alignement with headdirection_mean
        if np.dot(headvector,headdirection_mean[i]) > 0:
            pass
        else:
            # negative dot product points in opposite direction
            headvector = - headvector

        headvectors.append(headvector)
    
    return headcenter_med, np.array(headvectors), headdirection_mean

def projectionangle(center, vector, point):
    '''
    calculate the projection angle of a point on the head vector
    TODO troubleshoot
    '''
    
    # calculate reference vector from head center to point
    refvector = (point - center)/np.linalg.norm(point - center)
    vector = vector/np.linalg.norm(vector)

    # calculate projection angle between head vector and reference vector
    projectionangle = []
    for i in range(len(vector)):
        # TODO Error here somewhere
        try:
            test = refvector[i]
            projectionangle.append(np.arccos(np.dot(vector[i], refvector[i])))
        except:
            projectionangle.append(-9)

    return np.array(projectionangle)


def decomposeprojectionangle(center, headvector, point):
    '''
    calculate plane and elevation angles from head vector
    ''' 
    
    # calculate reference vector from head center to point
    refvector = (point - center)/np.linalg.norm(point - center)
    planerefvector = np.array([refvector[0], refvector[1], 0])/np.linalg.norm(np.array([refvector[0], refvector[1], 0]))
    elevationrefvector = np.array([0, 0, refvector[2]])/np.linalg.norm(np.array([0, 0, refvector[2]]))

    # separate head vector into plane and elevation vectors
    planevector = np.array([headvector[0], headvector[1], 0])/np.linalg.norm(np.array([headvector[0], headvector[1], 0]))
    elevationvector = np.array([0, 0, headvector[2]])/np.linalg.norm(np.array([0, 0, headvector[2]]))

    elevation = np.arccos(np.dot(elevationvector, elevationrefvector))
    azimuth = np.arccos(np.dot(planevector, planerefvector))

    return elevation, azimuth


def kinematicvector(pose, n = 25):
    '''
    calcualte the vector of movement of the body centroid in time window n
    '''

    # calculate the centroid of the body
    centroid = reduceposetocentroid(pose)
    
    # calculate the vector of movement of the body centroid in time window n
    kinematicvec = np.diff(centroid, n, axis=0, append = np.zeros(centroid.iloc[0:n,:].shape))

    # normalize kinematic vector
    kinematicvec = kinematicvec/np.linalg.norm(kinematicvec)
    
    return centroid, kinematicvec

def angularvelocity():
    '''calculate the angular velocity of the head'''

    return

def bodyvector():
    '''calculate the vector of body orientation from tail to neck'''
    
    return

def median_offset_filtering(pose, bodypart, k_tracking =  21, max_dist = 30, interpolation = 'linear'):
    '''
    Returns a filtered signal by removing and interpolating outliers from signal with distance to median filter > max_dist.
    pose: pandas DF from `behaviorspecific.read_anipose_data`
    bodypart: list of strings of body parts to track
    k_tracking: in frames, smoothing kernel
    max_dist: in mm, maximum distance between any given point on head
    '''
    # get raw coordinates
    signal = pose.loc[:, pose.columns.str.contains('|'.join(bodypart))]
    signal_x = signal.loc[:,signal.columns.str.contains('_x')]
    signal_y = signal.loc[:,signal.columns.str.contains('_y')]
    signal_z = signal.loc[:,signal.columns.str.contains('_z')]

    # calculate spatiotemporal reference

    ref_centroid = reduceposetocentroid(pose, bodypart, med_smooth = k_tracking)

    # interpolate missing values above offset thresdholod
    filt_signal_x = signal_x[abs(signal_x.subtract(ref_centroid.x, axis = 0)) < max_dist].interpolate(method=interpolation, order = 1, limit_direction = 'both')
    filt_signal_y = signal_y[abs(signal_y.subtract(ref_centroid.y, axis = 0)) < max_dist].interpolate(method=interpolation, order = 1, limit_direction = 'both')
    filt_signal_z = signal_z[abs(signal_z.subtract(ref_centroid.z, axis = 0)) < max_dist].interpolate(method=interpolation, order = 1, limit_direction = 'both')

    filtered_signal = pd.concat([filt_signal_x, filt_signal_y, filt_signal_z], axis = 1)
    
    return filtered_signal

def distance_to_plane(p1, p2, p3, point):
    '''
    Calculate the distance between a point and a plane.
    '''
    # vectors
    AB = np.subtract(p2, p1)
    AC = np.subtract(p3, p1)
    # normal to the plane
    normal = np.cross(AB, AC)
    # normalize the normal vector
    magnitude = np.linalg.norm(normal)
    normal = normal / magnitude
    # coefficients
    a, b, c = normal
    d = -np.dot(normal, p1)
    plane = [a, b, c, d]

    # distance to plane
    a = plane[0]
    b = plane[1]
    c = plane[2]
    x = point.iloc[:,0]
    y = point.iloc[:,1]
    z = point.iloc[:,2]
    distance = abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)
    
    return distance

def find_pecks(interval, fps):
    '''
    Find pecks in the distance to key signal.
    '''

    # CALCULATE GLOBAL reference
    k = int((len(interval)/10) // 2 * 2 + 1)
    # calculate global criterion
    global_criterion = scipy.signal.medfilt(np.pad(interval, (k, k), 'median'), kernel_size=k)
    global_criterion = global_criterion[k:-k]

    # CALCULATE LOCAL reference
    # get upper and lower contour lines
    upper_contour = pd.Series(0, index = interval.index).astype('float')
    lower_contour = pd.Series(0, index = interval.index).astype('float')
    maxima = scipy.signal.argrelextrema(interval.values, np.greater, order = 2) # left and right padding, total 100ms
    minima = scipy.signal.argrelextrema(interval.values, np.less, order = 2)
    upper_contour.loc[maxima[0]] = interval.loc[maxima[0]]
    lower_contour.loc[minima[0]] = interval.loc[minima[0]]
    upper_contour.replace(0, np.nan, inplace = True)
    lower_contour.replace(0, np.nan, inplace = True)
    upper_contour = upper_contour.interpolate(method='linear', order = 1, limit_direction = 'both')
    lower_contour = lower_contour.interpolate(method='linear', order = 1, limit_direction = 'both')
    median_contour = ((upper_contour + lower_contour) / 2).values

    # find and exclude valleys above local criterion
    dist = 0.200 *fps # expected inter-peck interval of 250-300ms, below that might be noise
    valleys, _ = scipy.signal.find_peaks(-interval, distance = dist, height = (-median_contour, None), prominence = (10, 100))
    valleys = valleys + interval.index.start
    # global criterion
    valleys = valleys[interval.iloc[valleys] < global_criterion[valleys]]
    # statistical criterion
    peckheight = interval.iloc[valleys].quantile(0.8) # is this too much?
    valleys = valleys[interval.iloc[valleys] < peckheight]
    
    return valleys

######################################
# TODO
### egocentric head orientation (relative to body direction)

# time series and angular speeds for head movements

# alignement or phase shifts between head direction and body direction

### FIELD of VIEW and projection angle
# extract peak locations
# how to test for statistical differences from distribution

# select head orientation responses by change in angular velocity of head direction 

# locomotion direction as angle between consecutive timepoints 
"""
Manze's single trial analysis tools
"""
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from visanalysis.util import plot_tools
from collections.abc import Sequence
import visanalysis.analysis.manalysis_ID as ImagingData

def computeRoiDffResponse(ImagingData, roi_name, full_trace_baseline=False):
    """
    Computes dF/F for a matrix of ROI responses by calling computeBaselineFittingDFF for each ROI

    Returns:
         roi_dff_data (similar to roi_data in getRoiResponses): dict, keys:
                        roi_dff: ndarry, shape = (rois, time)
                        time_vector: 1d array, time point value from acquision
                        roi_mask: list of ndarray masks, one for each roi in roi set
                        roi_image: ndarray image showing roi overlay
    """
    roi_data = ImagingData.getRoiResponses(roi_name)
    baseline_ind = ImagingData.getBaselineImageFrames(full_trace = full_trace_baseline)

    no_rois = len(roi_data.get('roi_response'))
    roi_dff = np.zeros_like(roi_data.get('roi_response'))
    time_vector = roi_data.get('time_vector')

    for r_ind in range(no_rois):
        roi_response = roi_data.get('roi_response')[r_ind][0]
        roi_dff[r_ind, :] = computeBaselineFittingDFF(roi_response, time_vector, baseline_ind)

    roi_dff_data = {
        'roi_dff': roi_dff,
        'time_vector': time_vector,
        'roi_mask': roi_data.get('roi_mask'),
        'roi_image': roi_data.get('roi_image')
    }

    return roi_dff_data

def computeBaselineFittingDFF(response, time_vector, baseline_img_frames):
    """
    Computes dF/F by fitting F0 using a sum of two exponentials to the baseline period

    Params:
        -response: 1D np.array of background subtracted single ROI response
        -time_vector: 1D np.array of time points corresponding to response
        -baseline_img_frames: list or array of frame indices for fitting baseline

    Returns:
        -dff_response: 1D np.array of dF/F response
    """
    
    def two_exp(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d * x)
    
    baseline_response = response[baseline_img_frames]
    baseline_time = time_vector[baseline_img_frames]
    initial_guess = [1, -0.1, 1, -0.01]

    try:
        popt, pcov = curve_fit(two_exp, baseline_time, baseline_response, p0=initial_guess, maxfev=10000)
        fitted_baseline = two_exp(time_vector, *popt)

    except:
        fitted_baseline = np.zeros_like(response)

    dff_response = (response - fitted_baseline) / (fitted_baseline + 1e-12)

    return dff_response

def getEpochResponseResampled(ImagingData, roi_name, epoch_index=None, resample_bin_frequency=120, dFF=True, full_trace_baseline=False):
    """
    Get epoch reponse matrix by resampling to a fixed bin frequency for each ROI. Need to indicate
    which epoches to 

    Params:
        -ImagingData: an ImagingData Object (a T-series) with ROI defined
        -roi_name: string, name of the group of ROIs
        -epoch_index: (int) 1D np.array, the index of the epoch of interest to average together, default None (all epochs)
        -resample_bin_frequency: frequency (in Hz) to resample the epoch responses to, default 120Hz

    Returns:
        -resampled_roi_response: dict with keys:
            -'resampled_average_epoch_response': 2D np.array of shape (no_rois, no_resampled_timepoints_in_an_epoch)
            -'time_vector': 1D np.array of time points corresponding to resampled_average_epoch_response

    """

    if dFF:
        roi_data = computeRoiDffResponse(ImagingData, roi_name, full_trace_baseline=full_trace_baseline)
    else:
        roi_data = ImagingData.getRoiResponses(roi_name, return_erm=False) # non-dFF function is not ready, because getRoiResponses['roi_response'] returns a list instead of a np.array
    
    imaging_time = roi_data.get('time_vector')
    if epoch_index is None:
        epoch_index = np.arange(len(ImagingData.getEpochParameters(param_key='stim_time')))

    stimulus_start_times = ImagingData.getStimulusTiming().get('stimulus_start_times')[epoch_index]
    stimulus_end_times = ImagingData.getStimulusTiming().get('stimulus_end_times')[epoch_index]
    real_stim_durations = stimulus_end_times - stimulus_start_times
    real_stim_time = np.median(real_stim_durations)
    #12112025 TODO: real stim time (end - start) are 0.024 or 0.008 (when specified as 0.02) 
    # - is that common or just because the specified value is not multiple of projector refresh rate? seems better in longer stimulus

    # use the shortest pre and tail time as a standard epoch
    pre_time = np.min(ImagingData.getEpochParameters(param_key='pre_time'))
    tail_time = np.min(ImagingData.getEpochParameters(param_key='tail_time'))
    
    #check if all stim_time are the same, if not, raise error, resampling cannot be done
    stim_times = np.array(ImagingData.getEpochParameters(param_key='stim_time'))[epoch_index]
    if not np.all(stim_times == stim_times[0]):
        raise ValueError('All stim_time for the selected epochs must be the same for resampling!')
    stim_time = stim_times[0]
    
    resample_fr = 1 / resample_bin_frequency
    no_bins_pre_stim = int(np.ceil(pre_time * resample_bin_frequency))
    no_bins_after_stim = int(np.ceil((stim_time + tail_time) * resample_bin_frequency))
    no_bins = no_bins_pre_stim + no_bins_after_stim

    pre_time_in_bin = no_bins_pre_stim * resample_fr
    after_time_in_bin = no_bins_after_stim * resample_fr

    # initialize resampled response matrix (is it legit to do zero or should it be nan?)
    resampled_roi_response = np.zeros((len(roi_data.get('roi_dff')), no_bins))
    resampled_sample_count = np.zeros((len(roi_data.get('roi_dff')), no_bins))
    resampled_time_vector = []

    # loop through each epoch to bin/resample to the resample_bin_frequency, using stim_start_time as reference
    for ep in range(len(epoch_index)):
        stim_present_time = stimulus_start_times[ep]
        start_time = stim_present_time - pre_time_in_bin
        end_time = stim_present_time + after_time_in_bin

        # get time vector for this epoch
        current_epoch_ind = np.where((imaging_time >= start_time) & (imaging_time < end_time))[0]
        current_epoch_time_vector = imaging_time[current_epoch_ind]
        current_epoch_response = roi_data.get('roi_dff')[:, 0, current_epoch_ind]

        # create new resampled time vector, the time stamp is the start of each bin 
        # (except for last bin which has both start and end time, because np.histogram need the rightmost edge)
        # so len(resampled_time_vector) = no_bins_pre_stim + no_bins_after_stim + 1
        resampled_epoch_time_vector = np.concatenate([np.linspace(stim_present_time - resample_fr * no_bins_pre_stim, stim_present_time, no_bins_pre_stim, endpoint=False), 
                                 np.linspace(stim_present_time, stim_present_time + resample_fr * no_bins_after_stim, no_bins_after_stim + 1, endpoint=True)]) 

        # bin the data
        binned_epoch_response = np.zeros((current_epoch_response.shape[0], no_bins_pre_stim + no_bins_after_stim))
        for roi in range(current_epoch_response.shape[0]):
            binned_epoch_response[roi, :], _ = np.histogram(current_epoch_time_vector, bins=resampled_epoch_time_vector, weights=current_epoch_response[roi, :])
            binned_epoch_sample_count, _ = np.histogram(current_epoch_time_vector, bins=resampled_epoch_time_vector)
        
        resampled_roi_response += binned_epoch_response
        resampled_sample_count += binned_epoch_sample_count
        resampled_time_vector = [resampled_time_vector, resampled_epoch_time_vector] # do we need this?

    # average across epochs
    resampled_average_epoch_response = resampled_roi_response / (resampled_sample_count + 1e-12)
    # 12112025 TODO: calculate resampled SEM for each roi each bin (then need to store individual responses), is that necessary?


    # unified in-epoch time vector
    in_epoch_time_vector = np.linspace(-pre_time_in_bin, after_time_in_bin, no_bins, endpoint=False) + resample_fr / 2  # center of each bin

    resampled_roi_response_dict = {
        'resampled_average_epoch_response': resampled_average_epoch_response,
        'time_vector': in_epoch_time_vector
    }  
    return resampled_roi_response_dict

def getMatchingEpochIndices(ImagingData, query):
    """
    Returns indices of epochs where parameters match the query.
    Handles float comparison safely.
    
    Params:
        ImagingData object.
        query (dict): Dictionary of parameters to match (e.g. {'intensity': 0.5}).
        
    Returns:
        list: Indices of epochs matching the query.
    """
    indices = []
    for i, params in enumerate(ImagingData.getEpochParameters()):
        match = True
        for key, value in query.items():
            param_value = params.get(key)
            if param_value is None:
                match = False
                break
            # Handle float comparison
            if isinstance(value, float) and isinstance(param_value, float):
                if not np.isclose(value, param_value):
                    match = False
                    break
            else:
                if value != param_value:
                    match = False
                    break
        if match:
            indices.append(i)
    return indices

def plotRoiResponses(ImagingData, roi_name):
    roi_data = ImagingData.getRoiResponses(roi_name, return_erm=True)

    fh, ax = plt.subplots(1, int(roi_data.get('epoch_response').shape[0]+1), figsize=(30, 2))
    [x.set_axis_off() for x in ax]
    # [x.set_ylim([-0.25, 1]) for x in ax]

    for r_ind in range(roi_data.get('epoch_response').shape[0]):
        time_vector = roi_data.get('time_vector')
        no_trials = roi_data.get('epoch_response')[r_ind, :, :].shape[0]
        current_mean = np.nanmean(roi_data.get('epoch_response')[r_ind, :, :], axis=0)
        current_std = np.nanstd(roi_data.get('epoch_response')[r_ind, :, :], axis=0)
        current_sem = current_std / np.sqrt(no_trials)

        ax[r_ind].plot(time_vector, current_mean, 'k')
        ax[r_ind].fill_between(time_vector,
                               current_mean - current_sem,
                               current_mean + current_sem,
                               alpha=0.5)
        ax[r_ind].set_title(int(r_ind))

        if r_ind == 0:  # scale bar
            plot_tools.addScaleBars(ax[r_ind], 1, 1, F_value=-0.1, T_value=-0.2)

def filterDataFiles(data_directory,
                    target_fly_metadata={},
                    target_series_metadata={},
                    target_roi_series=[],
                    target_groups=[],
                    pmt_channel=None,
                    quiet=False):
    """
    Searches through a directory of visprotocol datafiles and finds datafiles/series that match the search values
    Can search based on any number of fly metadata params or run parameters

    Params
        -data_directory: directory of visprotocol data files to search through
        -target_fly_metadata: (dict) key-value pairs of target parameters to search for in the fly metadata
        -target_series_metadata: (dict) key-value pairs of target parameters to search for in the series run (run parameters)
        -target_roi_series: (list) required roi_series names
        -target_groups: (list) required names of groups under series group
        -pmt_channel: (int) which pmt channel was used for imaging. this is default to single-channel series. 
            if more than one pmt gain is > 0, it is the "OR" logic. default is None, means no filtering based on pmt channel
                -0: red
                -1: green
                -2: far-red

    Returns
        -matching_series: List of matching series dicts with all fly & run params as well as file name and series number
    """
    fileNames = glob.glob(data_directory + "/*.hdf5")
    if not quiet:
        print('Found {} files in {}'.format(len(fileNames), data_directory))

    # collect key/value pairs for all series in data directory
    all_series = []
    for ind, fn in enumerate(fileNames):

        with h5py.File(fn, 'r') as data_file:
            for fly in data_file.get('Subjects'):
                fly_metadata = {}
                for f_key in data_file.get('Subjects').get(fly).attrs.keys():
                    fly_metadata[f_key] = data_file.get('Subjects').get(fly).attrs[f_key]

                for epoch_run in data_file.get('Subjects').get(fly).get('epoch_runs'):
                    series_metadata = {}
                    for s_key in data_file.get('Subjects').get(fly).get('epoch_runs').get(epoch_run).attrs.keys():
                        series_metadata[s_key] = data_file.get('Subjects').get(fly).get('epoch_runs').get(epoch_run).attrs[s_key]
                    acq = data_file.get('Subjects').get(fly).get('epoch_runs').get(epoch_run).get('acquisition')
                    series_metadata['pmt_0'] = float(acq.attrs.get('pmtGain_0', 0))
                    series_metadata['pmt_1'] = float(acq.attrs.get('pmtGain_1', 0))
                    series_metadata['pmt_2'] = float(acq.attrs.get('pmtGain_2', 0))

                    new_series = {**fly_metadata, **series_metadata}
                    new_series['series'] = int(epoch_run.split('_')[1])
                    new_series['file_name'] = fn.split('\\')[-1].split('.')[0]

                    existing_roi_sets = list(data_file.get('Subjects').get(fly).get('epoch_runs').get(epoch_run).get('rois').keys())
                    new_series['rois'] = existing_roi_sets
                    existing_groups = list(data_file.get('Subjects').get(fly).get('epoch_runs').get(epoch_run).keys())
                    new_series['groups'] = existing_groups

                    all_series.append(new_series)

    # search in all series for target key/value pairs
    match_dict = {**target_fly_metadata, **target_series_metadata}
    matching_series = []
    for series in all_series:
        if checkAgainstTargetDict(match_dict, series):
            if np.all([r in series.get('rois') for r in target_roi_series]):
                if np.all([r in series.get('groups') for r in target_groups]):
                    if pmt_channel is not None:
                        if series.get('pmt_{}'.format(pmt_channel)) > 0:
                            matching_series.append(series)
                    else:
                        matching_series.append(series)

    matching_series = sorted(matching_series, key=lambda d: d['file_name'] + '-' + str(d['series']).zfill(3))
    if not quiet:
        print('Found {} matching series'.format(len(matching_series)))
    return matching_series

def checkAgainstTargetDict(target_dict, test_dict):
    for key in target_dict:
        if key in test_dict:
            if not areValsTheSame(target_dict[key], test_dict[key]):
                return False  # Different values
        else:
            return False  # Target key not in this series at all

    return True

def areValsTheSame(target_val, test_val):

    if isinstance(target_val, str):
        return target_val.casefold() == test_val.casefold()
    elif isinstance(target_val, bool):
        if isinstance(test_val, str):
            return str(target_val).casefold() == test_val.casefold()

        return target_val == test_val

    elif isinstance(target_val, (int, float, np.integer, np.floating)):  # Scalar
        if isinstance(test_val, (int, float, np.integer, np.floating)):
            return float(target_val) == float(test_val)  # Ignore type for int vs. float here
        else:
            return False
    elif isinstance(target_val, (Sequence, np.ndarray)):  # Note already excluded possibility of string by if ordering
        if isinstance(test_val, (Sequence, np.ndarray)):
            # Ignore order of arrays, and ignore float vs. int
            return np.all(np.sort(target_val, axis=0).astype(float) == np.sort(test_val, axis=0).astype(float))
        else:
            return False

    else:
        print('----')
        print('Unable to match ')
        print('Target {} ({})'.format(target_val, type(target_val)))
        print('Test {} ({})'.format(test_val, type(test_val)))
        print('----')
        return False
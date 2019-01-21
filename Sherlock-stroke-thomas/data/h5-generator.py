# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:35:20 2018

@author: dumle
"""
from __future__ import division

import os
import sys

sys.path.append('../') #this allows us to import files form the ./stroke-deep-learning/src/-folder
import os
from os import listdir
os.chdir('/home/users/thomaslj/stroke-thomas')
# os.chdir(r'C:\Users\dumle\OneDrive\Skrivebord\Sherlock-stroke-thomas')
from utils import utilities
import h5py
import numpy as np
import pandas as pd
import sys

debugging = False
merged = True
revised_preprocessing = False
rescale_mode = 'soft'
cohort = 'SHHS'


if cohort == 'SHHS':
    epoch_duration = 5*60
    edf_folder = os.path.abspath('/home/users/thomaslj/stroke-thomas/data/shhs_edfs')
    # edf_folder = os.path.abspath(r'D:\9. semester (Stanford)\Cohorts\shhs\polysomnography\EDFs\Both')
    hypnogram_folder = None
    csv_path = os.path.abspath('/home/users/thomaslj/stroke-thomas/data/matched_controls.csv')
    # csv_path = os.path.abspath(r"C:\Users\dumle\OneDrive\Dokumenter\GitHub\stroke-deep-learning\matched_controls.csv")
    csv_ids = pd.read_csv(csv_path, dtype=str)
    healthy = csv_ids['conIDs']
    stroke = csv_ids['expIDs']
    group = np.concatenate([np.zeros(len(healthy)),
                           np.ones(len(stroke))])
    IDs = pd.concat((healthy, stroke), ignore_index=True)

    if merged:
        channels_to_load = {'eeg1': 0, 'eeg2': 1, 'ecg': 2, 'sao2': 3, 
                            'heartrate': 4, 'abdomen': 5, 'thorax': 6, 'airflow': 7}
        output_folder_s = os.path.abspath('/scratch/users/thomaslj/data/cohorts/shhs/stroke/h5')
        output_folder_h = os.path.abspath('/scratch/users/thomaslj/data/cohorts/shhs/healthy/h5')
        # output_folder_s = os.path.abspath('D:/9. semester (Stanford)/Cohorts/Test')
        # output_folder_h = os.path.abspath('D:/9. semester (Stanford)/Cohorts/Test')
        channel_alias = utilities.read_channel_alias(os.path.abspath('/home/users/thomaslj/stroke-thomas/data/merged_labels.json'))
        # channel_alias = utilities.read_channel_alias(os.path.abspath('insert marged_labels.json file path from local drive in debug'))
for counter, ID in enumerate(IDs):
    try:
        print('Processing: ' + str(ID[:-2]) + '(number ' + str(counter+1) + ' of ' + str(len(IDs)) + ').')
        if cohort == 'SHHS':
            filename = os.path.join(edf_folder , 'shhs1-' + ID[:-2] + '.edf')
        elif cohort == 'WSC':
            print('Dette er en prøve')
        try:
            data = utilities.load_edf_file(filename, channels_to_load, cohort = cohort,
                                           channel_alias = channel_alias, merged = merged)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            print('    EDF error (loading failed).')
            continue
        if data == -1:
            print('    Ignoring this subject due to different header setup.')
            continue
        # if revised_preprocessing:
        #     z = remove_artefacts(data['x'],data['fs'])
        #     x = utilities.butter_bandpass_filter(z, .3, 40, data['fs'], order=32)
        else:
            x = data['x']
            # filtering the channels: EEG1, EEG2 and ECG
            channels_to_eeg = [channels_to_load['eeg1'], channels_to_load['eeg2']]
            channel_to_ecg = channels_to_load['ecg']
            x_filt_eeg = utilities.butter_bandpass_filter(data['x'][channels_to_eeg, :], 0.3, 35, data['fs'], order=2)
            x_filt_ecg = utilities.butter_highpass_filter(data['x'][channel_to_ecg, :], 0.3, data['fs'], order=2)
            x[channels_to_eeg] = x_filt_eeg
            x[channel_to_ecg] = x_filt_ecg
            # quantile normalization on channels:
            x = utilities.rescale(x, rescale_mode)
        n = x.shape[1]
        epoch_samples = epoch_duration * data['fs']
        n_epochs = n // epoch_samples
        epoched = np.zeros([int(len(channels_to_load)), int(n_epochs), int(epoch_samples)])
        for i in range(len(channels_to_load)):
            epoched[int(i), :, :] = np.asarray(list(zip(*[iter(x[int(i),:])] * int(epoch_samples))))
        if False:
            print('dette er en bot')
        if cohort == 'SHHS':
            if group[counter] == 1: # ==1 is stroke
                output_file_name = os.path.join(output_folder_s, 'shhs1-' + ID[:-2] + '.hpf5')
                with h5py.File(output_file_name, "w") as f:
                    for k, v in channels_to_load.items():
                        dset = f.create_dataset(k, data=epoched[v,:,:], chunks=True)
                    f.attrs['ID'] = np.string_(ID[:-2])
                    f.attrs['fs'] = data['fs']
    #                f.create('fs', data=data['fs'])
                    f.attrs['group'] = group[counter]
            else:       # ==0 is healthy
                output_file_name = os.path.join(output_folder_h, 'shhs1-' + ID[:-2] + '.hpf5')
                with h5py.File(output_file_name, "w") as f:
                    for k, v in channels_to_load.items():
                        dset = f.create_dataset(k, data=epoched[v, :, :], chunks=True)
                    f.attrs['ID'] = np.string_(ID[:-2])
                    f.attrs['fs'] = data['fs']
                    f.attrs['group'] = group[counter]
                
            print('Actually processed {}'.format(filename))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        print('Error happened while processing: {}'.format(str(filename)))
            
print("All files processed.")
        
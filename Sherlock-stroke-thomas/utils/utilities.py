# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:46:20 2018

@author: dumle
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pyedflib
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.preprocessing import StandardScaler


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = butter(order, [low, high], btype = 'band', output='sos')
    return sos

def butter_highpass(lowcut, fs, order):
    nyq = 0.5*fs
    low = lowcut/nyq
    sos = butter(order, low, btype='highpass', output='sos')
    return sos

# This function takes data, filters it with bandpass and returns filtered signal
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order = order)
    y = sosfiltfilt(sos,data)
    return y

def butter_highpass_filter(data, lowcut, fs, order=4):
    sos = butter_highpass(lowcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

# This function rescales the input signal x to either 'standardscaler' or 'soft'
def rescale(x, rescale_mode):
    if rescale_mode == 'standardscaler':
        scaler = StandardScaler()
        new_x = scaler.fit_transform(x)
#'soft' sets new 5 and 95 quartiles to -1 and 1
    elif rescale_mode == 'soft':                            
        q5 = np.percentile(x, 5, axis=1)
        q95 = np.percentile(x, 95, axis=1)
        new_x = np.zeros(x.shape)
        for i in range(len(x)):
            new_x[i, :] = 2 * (x[i, :] - q5[i]) / (q95[i] - q5[i]) - 1
    return new_x


def read_channel_alias(fp):
    # Based on output from channel_label_identifier
    with open(fp) as f:
        data = json.load(f)
    alias = {}
    for chan in data['categories']:
        for a in data[chan]:
            alias[a] = chan
    return alias

def load_edf_file(filename, channels_to_load, cohort, channel_alias, merged):
    f = pyedflib.EdfReader(filename)
    labels = f.getSignalLabels()
    contained = {channel_alias[e]: i for (i, e) in enumerate(labels) if e in channel_alias}
    if not contained or len(contained) != len(channels_to_load):
        print(labels)
        print(contained)
        return -1
    
    fss = f.getSampleFrequencies()
    if cohort == 'SHHS':
        if merged == True:
            fs = fss[contained['ecg']]
            n = f.getNSamples()[contained['ecg']]
            X = np.zeros((len(channels_to_load), n))
    #lowcut = .3
    #highcut = 40.
    for chan_name, chan_idx_in_file in contained.items():
        # g = f.getPhysicalMaximum(chan_idx_in_file) / f.getDigitalMaximum(chan_idx_in_file)
        # x = g*f.readSignal(chan_idx_in_file)
        x = f.readSignal(chan_idx_in_file)
        if fss[chan_idx_in_file] != fs:
            time = np.arange(0, len(x) / fss[chan_idx_in_file], 1 / fs)
            t = np.arange(0, len(x) / fss[chan_idx_in_file], 1 / fss[chan_idx_in_file])
            F = interp1d(t, x, kind='linear', fill_value = 'extrapolate')
            x = F(time)
        X[channels_to_load[chan_name],:] = x
    data = {'x': X, 'fs': fs, 'labels': labels}
    return data


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args        
        
def logger(s):
    return print('{} | {}'.format(datetime.now(), s))

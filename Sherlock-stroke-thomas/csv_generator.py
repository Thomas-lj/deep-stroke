# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:57:08 2018

@author: dumle
"""
# this file reads h5 data and saves it to a csv file
import glob
import os
import sys
from random import seed
from random import shuffle

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def gather_dataframe(input_dir, ext):

    filelist_raw = glob.glob(os.path.join(input_dir, '*' + ext))

    list_fileID = []
    list_subjectID = []
    for f in filelist_raw:
        fileID = os.path.splitext(os.path.split(f)[1])[0]
        subjectID = int(fileID.split('-')[-1])
        list_fileID.append(fileID)
        list_subjectID.append(subjectID)

    df = pd.DataFrame(columns=['File', 'FileID', 'SubjectID', 'Fold1', 'Length', 'Group', 'Cohort']).fillna(0)
    df.File = filelist_raw
    df.FileID = list_fileID
    df.SubjectID = list_subjectID

    return df.sort_values(by=['FileID'])

def assign_length_of_subject_epochs(df):
    for idx, row in df.iterrows():
        with h5py.File(row['File'], 'r') as f:
            length_epochs = f['eeg1'][:].shape[0]
            label = f.attrs['group']
            df.loc[idx, 'Group'] = label
            df.loc[idx, 'Length'] = length_epochs
            print('processing subject number' + str(idx) + 'of' + str(len(df)))
    return df.reset_index(drop=True)

def assign_subjects_to_partitions(df, split=None):
    if not split:
        split = [0.80, 0.10, 0.10]
    unique_subjects = sorted(list(set(df['SubjectID'])))
#    n_subjects = len(unique_subjects)
    folds = [col for col in df if col.startswith('Fold')]
#    n_folds = len(folds)

    for fold in folds:
        seed(int(fold[-1]))
        shuffle(unique_subjects)
        #trainID, evalID, testID = np.split(unique_subjects, [int(split[0] * n_subjects), int((split[0] + split[1]) * n_subjects)])
        tmpID, testID = train_test_split(unique_subjects, stratify = df['Group'], test_size = split[1])
        tmp_labels = [df.loc[df['SubjectID'] == v, 'Group'].values[0] for v in tmpID]
        trainID, evalID = train_test_split(tmpID, stratify = tmp_labels, test_size = len(testID)/len(tmpID))
        for id in df['SubjectID']:
            if id in trainID:
                df.loc[df['SubjectID'] == id, fold] = 'train'
            elif id in evalID:
                df.loc[df['SubjectID'] == id, fold] = 'eval'
            elif id in testID:
                df.loc[df['SubjectID'] == id, fold] = 'test'
            else:
                print('No subset assignment for {}.'.format(id))
#    csv_path = os.path.join(input_dir, 'cohort_data')
#    df.to_csv(csv_path)
#    print('CSV file saved in directory:', csv_path)
    return df.reset_index(drop=True)
                
def csv_save(input_dir, df):
    if "shhs" in input_dir:
        csv_path = os.path.join(input_dir, 'csv_shhs.csv')
        df.Cohort = df.Cohort.fillna('shhs')
    if "wsc" in input_dir:
        csv_path = os.path.join(input_dir, 'csv_wsc' + '.csv')
        df.Cohort = df.Cohort.fillna('wsc')
    df.to_csv(csv_path)
    print('CSV file saved in:', csv_path)
















#l = []
#for idx, row in df.iterrows():
#    with h5py.File(row['File'], 'r') as f:
#        dset = {k: f[k] for k in ['eeg2']}
#        df.loc[idx, 'Length'] = [v.shape[0] for idx, v in enumerate(dset.values()) if idx == 0][0]
import os
import datetime
import pyedflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from utils import *


# saves raw data in .npy files located in tmp/
# returns path to the .npy files
def process_rawdata(db_folder):

    # this is a file containing the subjects details
    subject_details = pd.read_excel(os.path.join(db_folder, "SubjectDetails.xls"))
    # the number of records contained in the dataset
    n_records = len(subject_details)
    duration = subject_details['Study Duration (hr)'].min()*3600
    # we read one of the record so you know what you are dealing with
    # and what type of signals are available
    signals, signal_headers, header = pyedflib.highlevel.read_edf(os.path.join(db_folder,"ucddb002.rec"))
    signal_headers_table = pd.DataFrame(signal_headers)
    # number of signals available
    n_signals = len(signal_headers_table)
    # We set the sampling rate and will upsample to have homogeneous data. All signals will be of the same sample rate (easier once again for the preprocessing)
    sample_rate = signal_headers_table['sample_rate'].max()
    signal_len = int(sample_rate*duration)

    respevents = pd.read_fwf(os.path.join(db_folder,"ucddb002_respevt.txt"), widths=[10,10,8,9,8,8,6,7,7,5], skiprows=[0,1,2], skipfooter=1, engine='python', names=["Time", "Type", "PB/CS", "Duration", "Low", "%Drop", "Snore", "Arousal", "Rate", "Change"])
    respevents["Time"] = (pd.to_datetime(respevents["Time"]) - pd.to_datetime(subject_details.loc[0, "PSG Start Time"])).astype("timedelta64[s]")%(3600*24)
    respevents["Time"] = pd.to_timedelta(respevents["Time"], unit="s")

    # we store the raw signals for each record in this array
    raw_records = np.zeros((n_records, signal_len, n_signals), dtype="float32")

    # we store a binary variable for the labels (True = respiratory event, False = no respiratory event)
    raw_respevents = np.zeros((n_records, signal_len, 1), dtype="bool")

    # we go through the data and extract signals and labels. You can put this loop in a function
    # if you want.
    for entry in tqdm(os.scandir(db_folder)):
        rootname, ext = os.path.splitext(entry.name)
        study_number = rootname[:8].upper()
        if not study_number.startswith("UCDDB"):
            continue
        subject_index, = np.where((subject_details["Study Number"] == study_number))[0]
        if ext == ".rec":
            signals, signal_headers, header = pyedflib.highlevel.read_edf(entry.path)
            for sig, sig_hdr in zip(signals, signal_headers):
                try:
                    signal_index, = np.where((signal_headers_table["label"] == sig_hdr["label"]))[0]
                except ValueError:
                    if sig_hdr["label"] == "Soud":
                        signal_index = 7
                if sig_hdr["sample_rate"] != 128:
                    q = int(sample_rate//sig_hdr["sample_rate"])
                    sig = np.repeat(sig, q)
                    sig = sig[:signal_len]
                    raw_records[subject_index,:,signal_index] = sig.astype("float32")
        elif rootname.endswith("respevt"):
            respevents = pd.read_fwf(os.path.join(db_folder,rootname + ".txt"), widths=[10,10,8,9,8,8,6,7,7,5], skiprows=[0,1,2], skipfooter=1, engine='python', names=["Time", "Type", "PB/CS", "Duration", "Low", "%Drop", "Snore", "Arousal", "Rate", "Change"])
            respevents["Time"] = (pd.to_datetime(respevents["Time"]) - pd.to_datetime(subject_details.loc[subject_index, "PSG Start Time"])).astype("timedelta64[s]")%(3600*24)
            respevents["Time"] = pd.to_timedelta(respevents["Time"], unit="s")
            for _, event in respevents.iterrows():
                onset = int(sample_rate*event["Time"].total_seconds())
                offset = onset + int(sample_rate*event["Duration"])
                raw_respevents[subject_index, onset:offset] = 1

    save_data("tmp/", "raw_respevents.npy", raw_respevents)
    save_data("tmp/", "raw_records.npy", raw_records)

    return ("tmp/raw_records.npy", "tmp/raw_respevents.npy")

# prepare data into epochs of epoch_duration (seconds) 
# returns batch of size (708, 7680, 14) as we need a shape divisible by 32
def make_batch(dataset, sample_rate, epoch_duration):
    # Calculate the number of samples per epoch
    samples_per_epoch = sample_rate * epoch_duration
    samples_per_epoch = 32*math.ceil(samples_per_epoch/32)
    # Initialize a list to store segmented epochs
    segmented_epochs = []

    # Iterate over each record in the training set
    for record in dataset:
        # Calculate the total number of samples in the record
        total_samples = record.shape[0]

        # Calculate the number of epochs for this record
        num_epochs = int(total_samples // samples_per_epoch)
        # print(num_epochs)
        # Iterate over each epoch
        for i in range(num_epochs):
            # Calculate the start and end indices for this epoch
            start_index = int(i * samples_per_epoch)
            end_index = int(start_index + samples_per_epoch)
            # print(samples_per_epoch)
            # Extract the epoch from the record
            epoch = record[start_index:end_index, :]

            # If necessary, pad or truncate the epoch to ensure it has the correct length
            if epoch.shape[0] < samples_per_epoch:
                pad_width = ((0, samples_per_epoch - epoch.shape[0]), (0, 0))
                epoch = np.pad(epoch, pad_width, mode='constant', constant_values=0)
            elif epoch.shape[0] > samples_per_epoch:
                epoch = epoch[:samples_per_epoch, :]

            # Append the segmented epoch to the list
            segmented_epochs.append(epoch)

    # Convert the list of segmented epochs into a numpy array
    segmented_epochs = np.array(segmented_epochs)

    return segmented_epochs

# raw_records (num_records, signal_len, num_features)
# in its current form, we get 30% of total records for train,val,test 
# returns tuple of indexes for train, val and test records (axis = 0 of raw_records)
def train_test_split_records(records):

    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.10
    idx = np.arange(records.shape[0])
    seed = 42

    train_idx, test_idx = train_test_split(idx, random_state = seed,test_size = 1 - train_ratio)
    train_idx, test_idx = train_test_split(test_idx, random_state = seed,test_size = test_ratio/(test_ratio + validation_ratio))
    val_idx, test_idx = train_test_split(test_idx, random_state = seed,test_size = test_ratio/(test_ratio + validation_ratio))
    
    return(train_idx, val_idx, test_idx)
# data processing provided from notebook

import pyedflib
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

db_folder = "physionet.org/files/ucddb/1.0.0"

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
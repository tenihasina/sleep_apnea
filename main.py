from utils import load_raw_data, clear_raw_data
from data_processing import process_rawdata, make_batch, train_test_split_records

sample_rate = 128
epoch_duration = 60

path_records, path_events = process_rawdata()
raw_records, raw_respevents = load_raw_data(path_records, path_events)
train_idx, val_idx, test_idx = train_test_split_records(raw_records)

X_train = make_batch(raw_records[train_idx,:,:], sample_rate, epoch_duration)
Y_train = make_batch(raw_respevents[train_idx,:,:], sample_rate, epoch_duration)
X_val = make_batch(raw_records[val_idx,:,:], sample_rate, epoch_duration)
Y_val = make_batch(raw_respevents[val_idx,:,:], sample_rate, epoch_duration)
X_test = make_batch(raw_records[test_idx,:,:], sample_rate, epoch_duration)
Y_test = make_batch(raw_respevents[test_idx,:,:], sample_rate, epoch_duration)


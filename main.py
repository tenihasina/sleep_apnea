import keras_tuner as kt

from data_processing import *
from model import *
from utils import *
from data_augment import *

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
# remove rawrecords and events from memory to save RAM
clear_raw_data()

checkpoint_dir = os.path.join("/content/drive/MyDrive/Data/physionet.org/checkpoint_dir", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def hyperparam_tuning(X_train, Y_train, X_val, Y_val, epochs = 2):
    hp = kt.HyperParameters()

    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory=checkpoint_dir,
        project_name="tuner",
    )

    print(tuner.search_space_summary())
    tuner.search(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))

    models = tuner.get_best_models(num_models=2)
    best_model = models[0]

    print(best_model.summary())
    print(tuner.results_summary())
    return tuner

def fit_best_model(tuner, X_train, Y_train, ):
    
    checkpoint_full_train = os.path.join(
            "physionet.org/train_augment", 
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    # Fit with the augmented dataset.
    path_augment = data_augment(X_train, sample_rate)

    x_all = np.concatenate((X_train, np.load(path_augment)))
    y_all = np.concatenate((Y_train, Y_train))
    tuner_fit_predict(tuner, checkpoint_full_train, x_all, y_all, X_val)


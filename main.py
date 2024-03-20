import keras_tuner as kt

from data_processing import *
from model import *
from utils import *
from data_augment import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sample_rate = 128
epoch_duration = 60
db_folder = "physionet.org/files/ucddb/1.0.0"
checkpoint_dir = os.path.join("/content/drive/MyDrive/Data/physionet.org/checkpoint_dir", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# create batches of size (708, 7680, 14) for records and (708, 7680, 1) for targets
# saved in tmp/, might need to del them as you go if too heavy on RAM
# return tuple of variable size depending on wether you want train, val or test
def get_batch_from_raw_data(db_folder, sample_rate, epoch_duration, train, val, test = False):
    
    batches = ()
    path_records, path_events = process_rawdata(db_folder)
    raw_records, raw_respevents = load_raw_data(path_records, path_events)
    train_idx, val_idx, test_idx = train_test_split_records(raw_records)
    if train:
        X_train = make_batch(raw_records[train_idx,:,:], sample_rate, epoch_duration)
        Y_train = make_batch(raw_respevents[train_idx,:,:], sample_rate, epoch_duration)
        save_data("tmp", "X_train", X_train)
        save_data("tmp", "Y_train", Y_train)
        batches.append(X_train, Y_train)

    if val:
        X_val = make_batch(raw_records[val_idx,:,:], sample_rate, epoch_duration)
        Y_val = make_batch(raw_respevents[val_idx,:,:], sample_rate, epoch_duration)
        save_data("tmp", "X_val", X_val)
        save_data("tmp", "Y_val", Y_val)
        batches.append(X_val, Y_val)

    if test:
        X_test = make_batch(raw_records[test_idx,:,:], sample_rate, epoch_duration)
        Y_test = make_batch(raw_respevents[test_idx,:,:], sample_rate, epoch_duration)
        save_data("tmp", "X_test", X_test)
        save_data("tmp", "Y_test", Y_test)
        batches.append(X_test, Y_test)

    # remove rawrecords and events from memory to save RAM
    clear_raw_data()

    return batches

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

def fit_best_model(tuner, X_train, Y_train):
    
    checkpoint_full_train = os.path.join(
            "physionet.org/train_augment", 
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    # Fit with the augmented dataset.
    path_augment = data_augment(X_train, sample_rate)

    x_all = np.concatenate((X_train, np.load(path_augment)))
    y_all = np.concatenate((Y_train, Y_train))
    tuner_fit_predict(tuner, checkpoint_full_train, x_all, y_all)

# In this script, we make a short example where we re-traini the best model from a previous hyperparameter tuning using randomsearch on a subset
def main():
    X_train, Y_train, X_test, Y_test = get_batch_from_raw_data(db_folder, sample_rate, epoch_duration, train = True, val = False, test = True)
    model = Unet1D(backbone_name = "resnet18_1d", classes = 1, input_shape = (X_train.shape[1], X_train.shape[2]))
    model.compile(
        optimizer = 'adam',
        loss = 'binary_focal_crossentropy',
        metrics=['accuracy']    
    )
    model.fit(x=X_train, y=Y_train, epochs=10)
    predictions = model.predict(X_test)

    conf_matrix = confusion_matrix(
            Y_test.reshape(-1).astype(int), 
            predictions.reshape(-1).astype(int)
        )

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()

main()
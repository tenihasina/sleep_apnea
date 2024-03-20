# augmentation example
# from ecglib.preprocessing.preprocess import *
# from ecglib.preprocessing.composition import *
import numpy as np
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise, Shift
from utils import *

audio_augmenter = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=True),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    Shift(min_shift=-0.2, max_shift=0.2, p=0.5)
])

def bloc_permutation(signal, bloc_size=128):
    return np.random.shuffle(signal.reshape((-1, bloc_size)))
def permutation(records, feature_idx):
  return np.apply_along_axis(func1d = bloc_permutation, axis=1, arr=records[:,:,feature_idx])

# ecg_augmenter = EcgCompose(transforms=[
#     SumAug(leads=[0, 6, 11]), 
#     RandomConvexAug(n=4), 
#     OneOf(transforms=[ButterworthFilter(), IIRNotchFilter()], transform_prob=[0.8, 0.2])
# ], p=0.5)

# noise addition and pitch modification on the audio feature
# saved in tmp/
def data_augment(X_train, sample_rate):

    num_epochs, epoch_length, num_features = X_train.shape
    # Apply augmentation on each segmented epoch
    sound_idx = 7
    augmented_train = np.copy(X_train)
    augmented_train[:,:,sound_idx] = audio_augmenter(samples=augmented_train[:,:,sound_idx], sample_rate=sample_rate)
    # print(augmented_epochs[i,:,sound_idx].shape)
    print("done audio")
    # Check the shape of the augmented epochs
    # augmented_epochs[:,:,ecg_idx] = ecg_augmenter(augmented_epochs[:,:,ecg_idx]) 
    print("Shape of augmented epochs:", augmented_train.shape)
    save_data("tmp/","augmented_train.npy", augmented_train)

    return os.path.join("tmp/","augmented_train.npy")



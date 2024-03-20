import os
import numpy as np

def clear_raw_data():
    
    del raw_respevents
    del raw_records
    return None

def load_raw_data(p1, p2):
    return (np.load(p1), np.load(p2))

def save_data(path, name, data):
    if not os.path.file.exists(path):
        os.mkdir(path=path)
    np.save(os.path.join(path, name), data)

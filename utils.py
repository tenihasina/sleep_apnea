def clear_raw_data():
    
    del raw_respevents
    del raw_records
    return None

def load_raw_data(p1, p2):
    return (np.load(p1), np.load(p2))
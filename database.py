import numpy as np
from pathlib import Path

class Profile:
    def __init__(self, name, descriptors, mean_descriptor):
        self.name = name
        self.descriptors = descriptors
        self.mean_descriptor = np.mean(descriptors)

    def add_descriptor(descriptor):
        # add to descriptors and new mean

def read_database_file(filename):
    # a database is just an array of Profiles
    # returns an array of Profiles
    with open(filename, 'rb') as file:
        db = pickle.load(file)
        return db

def write_database_file(database, filename):
    with open(filename, 'wb') as f:
        pickle.dump(database, f, pickle.HIGHEST_PROTOCOL

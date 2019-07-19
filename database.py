import numpy as np
from pathlib import Path

class Profile:
    def __init__(self, name, descriptor):
        self.name = name
        self.descriptors = [descriptor]
        self.mean_descriptor = np.mean(self.descriptors, axis = 0)

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)
        self.mean_descriptor = np.mean(self.descriptors, axis = 0)

    def read_database_file(filename):
        # a database is just an array of Profiles
        # returns an array of Profiles
        with open(filename, 'rb') as file:
            db = pickle.load(file)
            return db

    def write_database_file(database, filename):
        with open(filename, 'wb') as f:
            pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)

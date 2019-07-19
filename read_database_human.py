import pickle
from database import Profile

from camera import take_picture #removing this makes it die for dumb reasons

from descriptors import add_to_database

filename = input("Database pathname?\n")

with open(filename, 'rb') as file:
    database = pickle.load(file)
    print(database.keys())

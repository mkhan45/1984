import pickle
from database import Profile

from camera import take_picture
import matplotlib.pyplot as plt

import optimized_descriptors as desc

mode = input("add or match?\n")

if mode == "match":
    filename = input("Database pathname?\n")

    with open(filename, 'rb') as file:
        database = pickle.load(file)

    picture = take_picture()

    names = desc.match(picture, database)

    print(names)
elif mode == "add":
    name = input("What is the name?\n")
    filename = input("Database pathname?\n")
    with open(filename, 'rb') as file:
        database = pickle.load(file)
    picture = take_picture()
    desc.add_to_database(name, database, picture)
import pickle
from database import Profile

from camera import take_picture
import matplotlib.pyplot as plt

from descriptors import add_to_database

database_name = input("Filename of already existing database? (n for new database)") 

if database_name == 'n':
    database = {}
else:
    with open(database_name, 'rb') as file:
        database = pickle.load(file)

img_array = take_picture()

# fig,ax = plt.subplots()
# ax.imshow(img_array)

face_name = input("What name?\n")

add_to_database(face_name, database, img_array)

print(database)

filename = input("What should the file be called?\n")

with open(filename, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)

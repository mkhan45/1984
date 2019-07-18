import pickle
from database import Profile

path = input("What is the path of the folder?\n")
database = read_from_mp3_folder(path)

filename = input("What should the file be called?\n")

def read_from_img_folder(path):
    """
    Reads folder of images into database

    Parameters
    ----------
    path : String
        global path to folder

    Returns
    -------
    Database
        database of songs

    Dependencies
    ------------
    Song to fingerprint
    append_database
    """

    database = []

    folder = Path(path)
    for file in folder.iterdir():
        if not file.is_dir() and file.suffix == ".jpg": #this might want more extensions idk
            print(file.stem)
            name = file.stem
            #make Profile and append to database
            database.append(Profile(name, descriptors))

    return database

filename = input("folder path?:\n")
database = read_from_img_folder(filename)

with open(filename, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)

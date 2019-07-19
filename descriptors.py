import numpy as np
from dlib_models import download_model, download_predictor
download_model()
download_predictor()

from dlib_models import load_dlib_models

# this loads the dlib models into memory. You should only import the models *after* loading them.
# This does lazy-loading: it doesn't do anything if the models are already loaded.
load_dlib_models()

from dlib_models import models

face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]

def match(pic, database): 
    """
    Takes a picture and tries to find a match.

    Parameters
    ----------
    pic: [np.array()]
        An image, converted into a np.array(). The camera module does this automatically.

    database: Dictionary{String name : Profile profile}

    Returns
    -------
    String
        A string containing the name of the match, or "no match" if there was no match.
    """

    detections = list(face_detect(pic)) 

    for i in detections:
        shape = shape_predictor(pic, detections[i])
        descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))
        #check if the descriptor is in a profile in the database and return the name associated with that profile


def add_to_database(name, database, *pics):
    """
    Adds a profile to the database with a name and descriptors for any pictures added alongside.
    If a profile already exists with the same name, adds the pictures to that profile.

    Parameters
    ----------
    name: [String]
        An input string containing the name of the person to be added.
    *pics: 1 or more [np.array]
        One or more pictures whose descriptors will be added to the profile specified by the name.
    database: Dictionary{String name : Profile profile}
    Returns
    -------
    None.
        Profiles will be updated on the spot. If a profile with the name specified doesn't exist,
        a new one will be created.
    """
    
    detections = []
    for pic in pics:
        detections.append(face_detect(pic))

    if name in database: #idk what the database is called
        database[name].append(*detections)
    else:
        database[name] = detections

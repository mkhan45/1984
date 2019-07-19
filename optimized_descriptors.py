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

    database: Dictionary 
        {String name : Profile profile}

    Returns
    -------
    List
        A list containing the names of people in the picture, or "no matches" if there were no matches.
    """

    threshold = 3 #some num: determine through experimentation
    detections = list(face_detect(pic))
    s_desc = list() #sample descriptors

    for i in detections:
        shape = shape_predictor(pic, detections[i])
        descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))
        s_desc.append(descriptor)

    s_desc = np.array(s_desc)
    d_desc = np.vstack(tuple(database.values().descriptors)) #database descriptors
    d_names = np.vstack(tuple(database.keys()))

    #compute Euclidean distances:
    distances = np.sum(s_desc**2, axis=1)[:, np.newaxis] + np.sum(d_desc**2, axis=1) - 2*np.dot(s_desc, d_desc.T) 
   
    min_idxs = np.argmin(distances, axis=1) #indices of minimum distance for each name
    min_dists = np.amin(distances, axis=1) #minimum distance for each name
    min_idxs = min_idxs[min_dists > threshold] #cutoff distances that are larger than threshold
    
    return d_names[min_idxs]

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
        database[name].descriptors.append(*detections)
    else:
        database[name].descriptors = detections

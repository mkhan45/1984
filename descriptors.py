import numpy as np
from dlib_models import download_model, download_predictor
download_model()
download_predictor()

from dlib_models import load_dlib_models

from database import Profile

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
    List
        A list containing the names of people in the picture, or "no matches" if there were no matches.
    """

    detections = list(face_detect(pic)) 

    threshold = 3 #some num
    names = []
    for i in detections:
        shape = shape_predictor(pic, detections[i])
        descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))
        distances = {}
        for k, v in database.items():
            arr = np.array(v.descriptors)
            distances[k] = np.sum(np.sqrt((arr - descriptor)**2))
        min_dist_name = min(distances, key=distances.get())
        if min_dist_name <= threshold:
            names.append(min_dist_name)
        
    return names

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
        database[name].descriptors.append(*detections)
    else:
        new_profile = Profile(name, detections)
        database[name] = new_profile

        '''
    Takes in list of face detections then returns list of descriptors for each detection
    
    Parameters
    ------------
    detections: list(face detect rectangles)
    
    four edges of face rectangle
    
    Returns
    ------------
    descriptors: list(numpy arrays)
    
    (128,) shape descriptor vectors
    
    '''
import numpy as np
from dlib_models import download_model, download_predictor
download_model()
download_predictor()

from dlib_models import load_dlib_models

from profile import Profile

# this loads the dlib models into memory. You should only import the models *after* loading them.
# This does lazy-loading: it doesn't do anything if the models are already loaded.
load_dlib_models()

from dlib_models import models

face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]

def pairwise_dists(x, y):
    """ Computing pairwise distances using memory-efficient
        vectorization.

        Parameters
        ----------
        x : numpy.ndarray, shape=(M, D)
        y : numpy.ndarray, shape=(N, D)

        Returns
        -------
        numpy.ndarray, shape=(M, N)
            The Euclidean distance between each pair of
            rows between `x` and `y`."""
    dot = x @ y.T
    dists = -2 * dot
    dists +=  np.sum(x**2)
    dists += np.sum(y**2)
    return  np.sqrt(dists)

def match(descriptor, database, threshold): 
    """
    Takes a picture and tries to find a match.

    Parameters
    ----------
    descriptor: [np.array()]
        A (128,) vector descibing each face

    database: Dictionary{String name : Profile profile}

    Returns
    -------
    List
        A list containing the names of people in the picture, or "no matches" if there were no matches.
    """
    
    print("this function is updated")

    #threshold = 1.5 #some num
    min_name = 'No Match Found'
    min_distance = threshold+.001
    for i, v in enumerate(database.values()):
        norm1 = v.mean_descriptor/np.linalg.norm(v.mean_descriptor)
        norm2 = descriptor/np.linalg.norm(descriptor)
        v_distance = np.sqrt(np.sum((norm1-norm2)**2))
        print(v_distance)
        if v_distance < min_distance:
            min_distance = v_distance
            min_name = v.name
    return min_name

        #check if the descriptor is in a profile in the database and return the name associated with that profile

def pic_to_detect(*pics):

    detections = []
    for pic in pics:
        detections.append(face_detect(pic))
    
    return detections

def detect_to_desc(detections):
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
    descriptors = []
    for i in detections:
        shape = shape_predictor(pic, detections[i])
        descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))
        descriptors.append(descriptor)

    return descriptors

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


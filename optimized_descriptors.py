import numpy as np
from sys import exit
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
    detections = pic_to_detect(pic)
    s_desc = np.array(detect_to_desc(pic, detections))
    if s_desc.size == 0:
        print("No face detected.")
        exit(0)
    print(s_desc)
    d_desc = np.vstack(tuple(val.mean_descriptor for val in database.values())) #database descriptors
    print(d_desc)
    d_names = np.vstack(tuple(database.keys()))

    #compute Euclidean distances:
    s_squared = np.sum(s_desc**2, axis=1)
    s_plus_d = s_squared[:, np.newaxis] + np.sum(d_desc**2, axis=1)
    distances = s_plus_d - 2*np.dot(s_desc, d_desc.T) 
   
    min_idxs = np.argmin(distances, axis=1) #indices of minimum distance for each name
    min_dists = np.amin(distances, axis=1) #minimum distance for each name
    min_idxs = min_idxs[min_dists > threshold] #cutoff distances that are larger than threshold
    
    return d_names[min_idxs]

def pic_to_detect(*pics):
    """
    Takes in any number of pictures and returns a list of detection rectangles

    Parameters
    ----------
    *pics: 1 or more [np.array]
        One or more pictures.

    Returns
    -------
    list
        A list of detection rectangles.
    """

    detections = face_detect(pics[0])
    for pic in pics[1:]:
        detections.append(face_detect(pic))
    
    return detections

def detect_to_desc(pic, detections):
    '''
    Takes in list of face detections then returns list of descriptors for each detection
    
    Parameters
    ------------
    detections: list
        Face detect rectangles
    
        four edges of face rectangle
    
    Returns
    ------------
    descriptors: list of numpy arrays
    
        (128,) shape descriptor vectors
    '''

    descriptors = []
    print(detections)
    for detection in detections:
        print("current detection: ")
        print(detection)
        shape = shape_predictor(pic, detection)
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
    
    descriptors = []
    for pic in pics:
        detection = (face_detect(pic))
        shape = shape_predictor(pic, detection)
        descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))
        descriptors.append(descriptor)

    if name in database: #idk what the database is called
        database[name].descriptors.append(*descriptors)
    else:
        database[name].descriptors = descriptors

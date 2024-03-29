import matplotlib.pyplot as plt
import pickle
from camera import take_picture
import numpy as np
from dlib_models import download_model, download_predictor, load_dlib_models
from dlib_models import models
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import optimized_descriptors as od
from our_profile import Profile
from descriptors import match
from pathlib import Path

download_model()
download_predictor()
load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]

def master_cam(databasepath = ''):
    '''
    Adds descriptors for faces in photo taken by camera to the requested database after asking for name
    
    Parameters
    ------------
    
    
    Returns
    ------------
    
    '''
    pickle_in = open(databasepath, "rb")
    database = pickle.load(pickle_in)
    
    
    fig,ax = plt.subplots()
    pic = take_picture()
    ax.imshow(pic)
    
    detections = list(face_detect(pic))
    descriptors = od.detect_to_desc(pic, detections)
    for i in range(len(descriptors)):
        match_name = match(descriptors[i], database, .4)
        if match_name == 'No Match Found':
            rect = detections[i]
            rectangle = patches.Rectangle((rect.left(),rect.bottom()),rect.right()-rect.left(),rect.top()-rect.bottom(),linewidth=1,edgecolor='r',facecolor='r', alpha=0.7)
            ax.add_patch(rectangle)
            rectangle.remove()
            name = input("What is the full name of this person?\n")
            if name in database.keys():
                database[name].add_descriptor(descriptors[i])
            else: database[name] = Profile(name, descriptors[i])
        else: 
            rect = detections[i]
            rectangle = patches.Rectangle((rect.left(),rect.bottom()),rect.right()-rect.left(),rect.top()-rect.bottom(),linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rectangle)
            ax.text(rect.left(), (rect.top()+rect.bottom())/2, match_name, color='r', fontsize = 16)
            database[match_name].add_descriptor(descriptors[i])
    with open(databasepath, 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
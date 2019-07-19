import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]
import optimized_descriptors as od
from database import Profile
from descriptors import match

database1 = dict()

from descriptors import match
def add_camera_pic(database = database1):
    '''
    Adds descriptors for faces in photo taken by camera to the requested database after asking for name
    
    Parameters
    ------------
    
    
    Returns
    ------------
    
    '''
    
    
    
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
            print(len(descriptors[i]))
            print(descriptors[i])
            database[match_name].add_descriptor(descriptors[i])

add_camera_pic()
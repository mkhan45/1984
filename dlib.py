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

detections = list(face_detect(pic)) #pic = take_picture() (in main.py)

for i in detections:
    shape = shape_predictor(pic, detections[i])
    descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))
    #add the descriptor to the group it belongs to, according to Whispers algorithm

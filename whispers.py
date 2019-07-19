import dlib_models
from dlib_models import load_dlib_models
load_dlib_models()

import node as nd
import numpy as np
import os
import pickle
import random
import skimage.io as io
from collections import Counter
from dlib_models import models
from pathlib import Path

IMG_EXT = [".jpg", ".jpeg", ".png"]
CUTOFF = 6  # The maximum Euclidean distance between descriptors to be neighbors
UPSCALE = 1  # The number of times to upscale the image before face detection.


descriptors = []  # a `numpy.ndarray` containing all the descriptors.
image_paths = []  # a list containing the file paths of each image.
graph = []  # a list of all the images as `Node` objects.
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]


folder = Path(input("What is the path to the image directory?\n> "))

# Populate `images` by converting each image into a descriptor.
for file in folder.iterdir():
    if not file.is_dir() and file.suffix in IMG_EXT:
        image = io.imread(file)
        if len(image.shape) == 3:
            image = image[..., :3]
        # Find the face and produce the descriptor.
        detection = list(face_detect(image, UPSCALE))
        if len(detection) == 0:
            continue
        detection = detection[0]
        shape = shape_predictor(image, detection)
        descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
        # add the image information to `descriptors` and `image_paths`
        descriptors.append(descriptor)
        image_paths.append(file)
descriptors = np.array(descriptors)


# Convert `descriptors` to a list of `Node` objects and store it in `graph`.
for i in range(len(descriptors)):
    # Calculate the Euclidean distance between descriptors.
    dist = np.sqrt(np.sum((descriptors - descriptors[i])**2, axis=1))
    # Determine the neighbors of the i-th node.
    neighbors = [id for id, dist in enumerate(dist) if id is not i and dist <= CUTOFF]
    # Create and add the node to `graph`
    graph.append(nd.Node(i, neighbors, descriptors[i], file_path=image_paths[i]))


rand_idxs = np.arange(len(graph))
print(rand_idxs)
for node_idx in rand_idxs:
    node = graph[node_idx]
    labels = Counter([graph[neighbor].label for neighbor in node.neighbors])
    most_common = max(labels.values())
    node.label = random.choice([label for label, freq in labels.items() if freq == most_common])

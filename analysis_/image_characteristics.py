""" This script extracts meta-data from images:
- bluriness (as variance of laplace)
- colour profile?
- location (NARS/THUL)
- Image dimensions """

###############################################################
########################## IMPORT PACKAGES ####################
###############################################################

from imutils import paths
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

###############################################################
########################## DEFINE FUNCTIONS ####################
###############################################################

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def location_scout(filename):
    if "NARS" in filename:
        location = "NARS"
    else:
        location = "THUL"
    return location

def colour_profiler(image):


def extract_meta_data(IMAGE_FOLDER):

    metadata = []

    for imagePath in paths.list_images(IMAGE_FOLDER):

        filename = imagePath.split(IMAGE_FOLDER, 1)[1] #extract filename from path
        
        image = cv2.imread(imagePath) #import image

        shape = image.shape #width, height and number of dimensions 

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to gray scale image (2 dims)

        fm = variance_of_laplacian(gray) #calculate blurriness score

        location = location_scout(filename)

        img_tupple = (filename, fm, location, shape)

        metadata.append(img_tupple)
    
    return(metadata)

###############################################################
########################## RUN SCRIPT ####################
###############################################################

IMAGE_FOLDER = "./images/"
my_metadata = extract_meta_data(IMAGE_FOLDER)



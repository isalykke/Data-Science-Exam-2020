""" This script extracts meta-data from images:
- bluriness (as variance of laplace)
- colour profile?
- location (NARS/THUL)
- Image dimensions """

from imutils import paths
import argparse
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def location_scout(filename):
    if "NARS" in filename:
        location = "NARS"
    else:
        location = "THUL"
    return location



ap =argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help= "path to input images")
args = vars(ap.parse_args())


for imagePath in paths.list_images(args["images"]):
    #load image, convert to gray and compute laplacian
    image = cv2.imread(imagePath)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(grey)
    text = "score"

    cv2.putText(image, f"{text}: {fm}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)



IMAGE_FOLDER = "./images/"

def extract_meta_data(IMAGE_FOLDER):

    metadata = []

    for imagePath in paths.list_images(IMAGE_FOLDER):

        filename = imagePath.split(IMAGE_FOLDER, 1)[1]

        image = load_img(imagePath)

        width, height = image.size
        
        image_for_laplace = cv2.imread(imagePath)
        gray = cv2.cvtColor(image_for_laplace, cv2.COLOR_BGR2GRAY) #convert to gray scale image (2 dims)
        fm = variance_of_laplacian(gray) #calculate blurriness score

        location = location_scout(filename)

        width, height = image.size

        img_tupple = (filename, fm, location, width, height)

        metadata.append(img_tupple)
    
    return(metadata)



my_metadata = extract_meta_data(IMAGE_FOLDER)



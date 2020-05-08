""" This script extracts meta-data from images:
- bluriness (as variance of laplace)
- colour profile?
- location (NARS/THUL)
- Image dimensions """

from imutils import paths
import argparse
import cv2


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

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





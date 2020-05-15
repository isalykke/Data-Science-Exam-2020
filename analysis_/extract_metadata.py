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
from scipy.integrate import quad
import pandas as pd
import sympy as sym
import math
import os

###############################################################
########################## DEFINE FUNCTIONS ####################
###############################################################

def location_scout(filename):
    if "NARS" in filename:
        location = "NARS"
    else:
        location = "THUL"
    return location

def label_false_positives(path):
    if "false_pos" in path:
        label = "1"
    else:
        label = "0"
    return label

def create_L_boxes(img, L):
    #creates non-overlapping boxes of size LxL
    L_boxes = []

    #loop over pixels in image to create L-boxes:
    for r in range(0, img.shape[0], L):
        for c in range(0, img.shape[1], L):
            L_box = img[r:r+L,c:c+L]
            L_boxes.append(L_box)

    return L_boxes

def calcuate_box_variance(box, mean_gb):
    gi_variance =[]

    for gi in box.flat:

        variance = (gi - mean_gb)**2

        gi_variance.append(variance)

    box_variance = np.sum(gi_variance)

    return box_variance

def resize_image(img, resize_scales, L):
    resized_images = []

    for scale in resize_scales:
        resized_width = int(img.shape[1] * scale /100) #calculate the downscaled width and height, keeping aspect ratios constant
        resized_height = int(img.shape[0] * scale /100)

        #making sure the image can be devided into LxL boxes
        if resized_width%L == 1:
            resized_width += -1
        if resized_height%L == 1:
            resized_height += -1
        
        dims = (resized_height, resized_width)

        resized_image = cv2.resize(img, dims, cv2.INTER_NEAREST) #resize the image using nearest neighbour
        
        resized_images.append(resized_image) #add to list
    
    return resized_images

def gray_level_mean_variance(normalized_gray, resize_scales, L):

    V_of_S = []

    resized_images = resize_image(normalized_gray, resize_scales, L)

    #divide images into L_boxes and calculate mean variances
    for i, img in enumerate(resized_images): #NB! why does this not work when I only run over one image?

        L_boxes = create_L_boxes(img, L) #creates roi-boxes ("L-boxes") of size LxL

        for box in L_boxes:

            Vbs = []

            #calculate the mean grey level, mean_gb, of the box
            box_sum = cv2.sumElems(box)
            mean_gb = (1/L**2)*box_sum[0] #eq 2
            #print(f"mean gb={mean_gb}")

            #calculate the sample variance of the gray level, Vb, of the box
            box_variance = calcuate_box_variance(box, mean_gb)
            Vb = (1/(L**2-1))*box_variance #eq 1
            #print(f"Vb={Vb}")
            Vbs.append(Vb)

        #calculate gray-level mean variance V, over all boxes:
        V = (L**2/(img.shape[0]*img.shape[1]))*np.sum(Vbs) #eq 3

        S = L*normalized_gray.shape[0]/img.shape[0] #eq 4

        V_of_S.append((V,S, resize_scales[i])) 

    return V_of_S

def Q_complexity(glmv):

    v = [math.log(i[0]) for i in glmv]
    s = [math.log(i[1]) for i in glmv]

    #find limits for integral
    Smax = max(s)
    Smin = min(s)

    v, s = sym.symbols("v s")
    f = S/V

    derivative = sym.diff(f, s)

    interval = (1-0.25)*derivative**2

    #the ramp function 
    if interval >= 0:
        my_expression = interval
    else:
        my_expression = 0

    integral = quad(lambda s: my_expression, Smin, Smax)

    Q = (1/(Smax-Smin))*integral[0] #eq5            #https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    #result = quad(lambda x: x**2, 0, 2) 


def extract_meta_data(data_folder):

    metadata = []

    for dirpath, subdirs, files in os.walk(data_folder):
        for file in files[1:6]:
            path = os.path.join(dirpath,file)
            print(path)

            filename = file #extract filename from path
            
            image = cv2.imread(path) #import image

            shape = image.shape #height, width and number of dimensions

            size =  shape[0] * shape[1] #height * width

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to gray scale image (2 dims)

            normalized_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            glmv = gray_level_mean_variance(normalized_gray, (5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95), 2)

            #complexity = Q_complexity(glmv)

            blur = cv2.Laplacian(gray, cv2.CV_64F).var() #calculate blurriness score

            location = location_scout(filename)

            label = label_false_positives(path)

            img_tupple = (filename, shape[0], shape[1], size, blur, location, label)

            metadata.append(img_tupple)
    
    return(metadata)

###############################################################
########################## RUN SCRIPT ######################
###############################################################

data_folder = "./data/"
my_metadata = extract_meta_data(data_folder)

#convert to csv file 
metadata = pd.DataFrame(my_metadata, columns = ["filename", "height", "width", "size", "blur", "location", "false_pos"])
metadata.to_csv('metadata.csv')



###############################################################
########################## PLOT IMAGES ####################
###############################################################


from matplotlib import pyplot 
data = pyplot.imread("./images/NARS-13_000074.JPG_4966.0_867_5247.0_1156.0.jpg")
# plot the image
pyplot.imshow(im)

image = cv2.imread("./images/NARS-13_000074.JPG_4966.0_867_5247.0_1156.0.jpg")
cv2_imshow(normalized_gray)



from IPython import display
from PIL import Image
def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(Image.fromarray(a))

resize_scales = (5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95)
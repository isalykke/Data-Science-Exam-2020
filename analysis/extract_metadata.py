"""
This is a script for extracting meta-data from images and exporting it as a .csv file
It is developed for my datascience exam at AU, spring 2020
It extracts:
- filename
- height, width and size of image
- bluriness (as variance of laplacian)
- a label for the image (positive/false positive)
- a complaxity measure implemented from "Zanette, D. H. (2018). Quantifying the complexity of black-and-white images. PloS one, 13(11)."

"""

__author__ = 'Isa Lykke Hansen'
__email__ = 'i@lykkeh.dk'


###############################################################
########################## IMPORT PACKAGES ####################
###############################################################

from imutils import paths
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from scipy.integrate import quad
from jax import grad
import pandas as pd
import math
import os
import re
from scipy.interpolate import BSpline
from scipy import interpolate

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
    if "false" in path:
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
        
        dims = (resized_width, resized_height) #dimensions need to be switched for cv2.resize jf. https://stackoverflow.com/questions/53277833/why-is-opencv-switching-the-shape-when-resizing

        resized_image = cv2.resize(img, dims, cv2.INTER_NEAREST) #resize the image using nearest neighbour
        
        resized_images.append(resized_image) #add to list
    
    return resized_images

def gray_level_mean_variance(normalized_gray, resize_scales, L):

    V_of_S = []

    resized_images = resize_image(normalized_gray, resize_scales, L)

    #divide images into L_boxes and calculate mean variances
    for i, img in enumerate(resized_images): #NB! why does this not work when I only run over one image?

        L_boxes = create_L_boxes(img, L) #creates roi-boxes ("L-boxes") of size LxL

        Vbs = []

        for box in L_boxes:

            #calculate the mean grey level, mean_gb, of the box
            box_sum = cv2.sumElems(box)
            mean_gb = (1/L**2)*box_sum[0] #eq 2
            #print(f"mean gb = {mean_gb}")

            #calculate the sample variance of the gray level, Vb, of the box
            box_variance = calcuate_box_variance(box, mean_gb)
            #print(f"box var = {box_variance}")
            Vb = (1/(L**2-1))*box_variance #eq 1
            #print(f"Vb = {Vb}")
            Vbs.append(Vb)

        #calculate gray-level mean variance V, over all boxes:
        V = ((L**2)/(img.shape[0]*img.shape[1]))*np.sum(Vbs) #eq 3

        #print(f"V = {V}")

        S = L*normalized_gray.shape[0]/img.shape[0] #eq 4

        V_of_S.append((V,S, resize_scales[i])) 

    return V_of_S

def Q_complexity(glmv):

    #take log of V and S
    v = [math.log(i[0]) for i in glmv]
    s = [math.log(i[1]) for i in glmv]

    #find limits for integral
    Smax = max(s)
    Smin = min(s)

    #reverse to increasing order for Bspline()
    v.reverse()
    s.reverse() 

    #convert to arrays for splrep
    x = np.array(s)
    y = np.array(v)

    #find vector knots for Bspline
    t, c, k = interpolate.splrep(x=x, y=y, s= 0, k=4)
    spl = BSpline(t, c, k)
    
    deriv = BSpline.derivative(spl)

    #s = [i for i in s if deriv(i)**2 <= 4]

    integral = quad(lambda s: (1-0.25)*deriv(s)**2, Smin, Smax)

    Q = (1/(Smax-Smin))*integral[0] #eq5   

    return Q

def ICLS_complexity(img, path, compression):

    uncomp_size = os.path.getsize(path)

    temp_compressed = cv2.imwrite(f'temp_compres{compression}.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), compression])

    comp_size = os.path.getsize(f'temp_compres{compression}.jpg') #size of temp compressed image

    os.remove(f'temp_compres{compression}.jpg') #remove the temp image from disk

    CR = uncomp_size/comp_size #eq1

    ICLS = 1/CR #eq2

    return ICLS

def extract_meta_data(data_folder):

    metadata = []

    for dirpath, _, files in os.walk(data_folder):
        for i,file in enumerate(files): 
            if file.endswith(".jpg"):

                path = os.path.join(dirpath,file)
                #print(f"{dirpath} {i}/{len(files)}")
                print(path)

                filename = file #extract filename from path

                img_no = re.findall('[\d]+(?=.JPG)',filename)[0] #extract image number from filename

                location = location_scout(filename)

                label = label_false_positives(path)
                
                image = cv2.imread(path) #import image

                shape = image.shape #height, width and number of dimensions

                size =  shape[0] * shape[1] #height * width

                ratio = shape[0] / shape[1] #height/width

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to gray scale image (2 dims)

                blur = cv2.Laplacian(gray, cv2.CV_64F).var() #calculate blurriness score

                normalized_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                ICLS10 = ICLS_complexity(image, path, 10)

                ICLS30 = ICLS_complexity(image, path, 30)

                ICLS50 = ICLS_complexity(image, path, 50)

                glmv = gray_level_mean_variance(normalized_gray, (3,4,5,7,10,12,15,17,20,25,30,40,50,60,70,80,90), 2)

                Q = Q_complexity(glmv)

                img_tupple = (filename, img_no, shape[0], shape[1], size, ratio, blur, location, label, ICLS10, ICLS30, ICLS50, Q)

                metadata.append(img_tupple)
    
    return(metadata)

###############################################################
########################## RUN SCRIPT ######################
###############################################################

resize_scales = (2.5,3,4,5,7,10,12,15,17,20,25,30,40,50,60,70,80,90)
data_folder = "./data/"
my_metadata = extract_meta_data(data_folder)

#convert to csv file and save
metadata = pd.DataFrame(my_metadata, columns = ["filename", "img_no", "height", "width", "size", "ratio", "blur", "location", "false_pos", "ICLS10", "ICLS30", "ICLS50", "Q"])
metadata.to_csv('metadata.csv')


###############################################################
########################## PLOT IMAGES ####################
###############################################################

from IPython import display
from PIL import Image
from matplotlib import pyplot as plt

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

image = cv2.imread("./images/g360.bmp")
cv2_imshow(image)

imggg = Image.open("./images/g360.bmp")


######################################plot B-splines and V(S)###########################################
#NB you need to have created a glmv-object with the function gray_level_mean_variance() for this to work:

#take log of V and S
v = [math.log(i[0]) for i in glmv]
s = [math.log(i[1]) for i in glmv]

#reverse order for Bspline()
v.reverse()
s.reverse() 

#convert to array for splrep
x = np.array(s)
y = np.array(v)

#find vector knots for Bspline
t, c, k = interpolate.splrep(x=x, y=y, s= 0, k=4)
spl = BSpline(t, c, k)

#plot the Bspline
print('''\
    t: {}
    c: {}
    k: {}
    '''.format(t, c, k))
N = 100
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)

plt.plot(x, y, 'bo', label='Original points')
plt.plot(xx, spline(xx), 'r', label='BSpline')
plt.grid()
plt.legend(loc='best')
plt.show()

#plot V(S)

    V = [i[0] for i in glmv]
    S = [i[1] for i in glmv]

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(S,V)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks((1,10,100))
    ax.set_yticks((1,10,100, 1000))
    plt.xlabel("S")
    plt.ylabel("V")
    plt.title('Ordered')
    ax.legend() 
    #fig.savefig("ordered_norm.png")
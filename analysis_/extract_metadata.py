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

def variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def location_scout(filename):
    if "NARS" in filename:
        location = "NARS"
    else:
        location = "THUL"
    return location

def label_false_positives(filename):
    if "false_positives" in filename:
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

def resize_image(img, resize_scales):
    resized_images = []

    for scale in resize_scales:
        resized_width = int(img.shape[1] * scale /100) #calculate the downscaled width and height, keeping aspect ratios constant
        resized_height = int(img.shape[0] * scale /100)

        dims = (resized_width, resized_height)

        resized_image = cv2.resize(img, dims, cv2.INTER_NEAREST) #resize the image using nearest neighbour

        resized_images.append(resized_image) #add to list
    
    return resized_images

def gray_level_mean_variance(nomalized_gray, resize_scales, L):
    
    #test variables: #####
    #L=2
    #normalized_gray = gray #for now. actually we need to normalize gray  first
    #resize_scales = (10,20,30,40,50,70,90) 
    #######################

    resized_images = resize_image(normalized_gray, resize_scales)

    #divide images into boxes of size LxL to calculate mean variances
    for img in resized_images[2]: #NB! why does this not work when I only run over one image?

        L_boxes = create_L_boxes(img, L) #creates roi-boxes ("L-boxes") of size LxL

        for i, box in enumerate(L_boxes):
            print(i)
            Vbs = []

            #calculate the mean grey level, mean_gb, of the box
            box_sum = cv2.sumElems(box)
            mean_gb = (1/L**2)*box_sum[0] #eq 2
            print(f"mean gb={mean_gb}")

            #calculate the sample variance of the gray level, Vb, of the box
            box_variance = calcuate_box_variance(box, mean_gb)
            Vb = (1/(L**2-1))*box_variance #eq 1
            print(f"Vb={Vb}")
            Vbs.append(Vb)

        #calculate gray-level mean variance V, over all boxes:
        V = (L**2/(img.shape[0]*img.shape[1]))*np.sum(Vbs) #eq 3

        #assign each V to a scale, S

    return V

#def Q_complexity(nomalized_gray):





def extract_meta_data(IMAGE_FOLDER):

    metadata = []

    for imagePath in paths.list_images(IMAGE_FOLDER):

        print(imagePath)

        filename = imagePath.split(IMAGE_FOLDER, 1)[1] #extract filename from path
        
        image = cv2.imread(imagePath) #import image

        shape = image.shape #height, width and number of dimensions

        size =  shape[0] * shape[1] #height * width

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to gray scale image (2 dims)

        #normalized_gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        blur = variance_of_laplacian(gray) #calculate blurriness score

        location = location_scout(filename)

        label = label_false_positives(filename)

        img_tupple = (filename, shape[0], shape[1], size, blur, location, label)

        metadata.append(img_tupple)
    
    return(metadata)

###############################################################
########################## RUN SCRIPT ####################
###############################################################

IMAGE_FOLDER = "./images/"
my_metadata = extract_meta_data(IMAGE_FOLDER)

my_metadata


""" maybe use this for cycling through directories:
import os
rootdir = 'C:/Users/sid/Desktop/test'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file)) """



from matplotlib import pyplot 
data = pyplot.imread("./images/NARS-13_000074.JPG_4966.0_867_5247.0_1156.0.jpg")
	# plot the image
pyplot.imshow(resized_images[0])


image = cv2.imread("./images/NARS-13_002067.JPG_4830.0_3294_5009.0_3409.0.jpg")
image.shape
cv2.imshow("image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)

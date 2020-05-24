import keras
from keras.preprocessing.image import ImageDataGenerator



randomizer:

loop through two directories (true/false)

generate random number btw 0 and 1
if number =< 0.2: #we do a 20/80 test/train split
    if folder = true
        put in test_folder_true:
    else:
        put in test folder_false
else put in train folder:
    if folder = true
        put in train_folder_true:
    else:
        put in train folder_false



#path to images
train_dir = 'CNN/train'
val_dir ='CNN/valid'

#The ImageDataGenerator class generates batches of tensor image data with real-time data augmentation/normalization 
#The data will be looped over (in batches).


augs_gen = ImageDataGenerator(
    rescale = 1./255, #rescaling factor. multiply the data by the value provided (after applying all other transformations)
    shear_range = 0.2,  #Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    zoom_range = 0.2,  #Range for random zoom.
    horizontal_flip = True, #Randomly flip inputs horizontally.
    validation_split = 0.2)  #Fraction of images reserved for validation (strictly between 0 and 1).


#flow_from_directory takes the dataframe and the path to a directory and generates batches of augmented/normalized data.
train_gen = augs_gen.flow_from_directory(
    train_dir, #directory to loop through
    target_size = (150,150), #The dimensions to which all images found will be resized.
    batch_size = 2, #the 2 different classes
    class_mode = 'categorical', #Determines the type of label arrays that are returned: "categorical" will be 2D one-hot encoded labels
    shuffle = True #default
)

#do the same for the test data
test_gen = augs_gen.flow_from_directory(
    val_dir,
    target_size=(150,150), 
    batch_size=2,
    class_mode='categorical',
    shuffle=False
)

num_classes = 2
epochs = 20
input_shape = (150, 150, 3) #width, height and RGB values


#create a model with some layers
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, SeparableConv2D, \
    BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Lambda(lambda x: x, input_shape = input_shape))

model.add(Conv2D(32,(3,3), activation='selu'))
model.add(SeparableConv2D(32, (3, 3), activation='selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#the following layers could be included, but only raises accuracy slightly
'''
model.add(Conv2D(64,(3,3),activation='selu'))
model.add(SeparableConv2D(64, (3, 3), activation='selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(128,(3,3),activation='selu'))
model.add(SeparableConv2D(128, (3, 3), activation='selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(256,(3,3),activation='selu'))
model.add(SeparableConv2D(256, (3, 3), activation='selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
'''

model.add(Flatten())
model.add(Dense(1024,activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))


from keras.optimizers import SGD
opt = SGD(lr=1e-4, momentum=0.99) #this will give you bang for your bucks

model.compile(loss = 'binary_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])
model.summary()

history = model.fit_generator(
    train_gen,
    steps_per_epoch = 50, #100
    epochs = 20, #20
    verbose = 1,
    #use_multiprocessing = True,
    validation_data= test_gen,
    validation_steps = 50 #100
)

score = model.evaluate(test_gen, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#how to save the model: https://machinelearningmastery.com/save-load-keras-deep-learning-models/

#######################################
#plot model performance:
#######################################

#plot model performance. from: https://www.kaggle.com/ruslankl/brain-tumor-detection-v1-0-cnn-vgg-16/comments
%matplotlib inline 
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()


#######################################
#predict the class of a new image:
#######################################

from keras.preprocessing.image import load_img, img_to_array

new_img = load_img("LEGO_brick_images/new/201706171606-0010.png", target_size=(150,150))
new_img = img_to_array(new_img) #convert to numpy array
new_img = new_img.reshape((1,) + new_img.shape) #add a 4th dimension for the model

# make a prediction on new image. from https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
ynew = model.predict_classes(new_img)
ynew

#show which class names to compare 
test_gen.class_indices
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from cv2 import os
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS= 25
INIT_LR = 1e-3
BS = 32
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
	# extract the class label from the image path and update the
	# labels list
    label = imagePath.split(os.path.sep)[-2]
    if label == "alfa":
        label = 0
    elif label == "audi":
        label = 1
    elif label == "bmw":
        label = 2
    elif label == "chevrolet":
        label = 3
    elif label == "citroen":
        label = 4
    elif label == "dacia":
        label = 5
    elif label == "daewoo":
        label = 6
    elif label == "dodge":
        label = 7
    elif label == "ferrari":
        label = 8
    if label == "fiat":
        label = 9
    elif label == "ford":
        label = 10
    elif label == "honda":
        label = 11
    elif label == "hyundai":
        label = 12
    elif label == "jaguar":
        label = 13
    elif label == "jeep":
        label = 14
    elif label == "kia":
        label = 15
    elif label == "lada":
        label = 16
    elif label == "lancia":
        label = 17
    if label == "landrover":
        label = 18
    elif label == "lexus":
        label = 19
    elif label == "masserati":
        label = 20
    elif label == "mazda":
        label = 21
    elif label == "mercedes":
        label = 22
    elif label == "mitshubisi":
        label = 23
    elif label == "nissan":
        label = 24
    elif label == "opel":
        label = 25
    elif label == "pegueot":
        label = 26
    if label == "porsche":
        label = 27
    elif label == "renault":
        label = 28
    elif label == "rover":
        label = 29
    elif label == "saab":
        label = 30
    elif label == "seat":
        label = 31
    elif label == "skoda":
        label = 32
    elif label == "subaru":
        label = 33
    elif label == "suzuki":
        label = 34
    elif label == "tata":
        label = 35
    if label == "tesla":
        label = 36
    elif label == "toyota":
        label = 37
    elif label == "volkswagen":
        label = 38
    elif label == "volvo":
        label = 39
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=40)
testY = to_categorical(testY, num_classes=40)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=40)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

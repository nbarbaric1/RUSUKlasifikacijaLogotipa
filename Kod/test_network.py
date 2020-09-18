# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()
# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)  
# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])
# classify the input image
(alfa, audi, bmw, chevrolet, citroen, dacia, daewoo, dodge, ferrari, fiat, ford, honda, hyundai, jaguar, jeep, kia, lada, lancia, landrover, lexus, masserati, mazda, mercedes, mitshubisi, nissan, opel, pegueot, porsche, renault, rover, saab, seat, skoda, subaru, suzuki, tata, tesla, toyota, volkswagen, volvo) = model.predict(image)[0]

# build the label
maxLabel = max(alfa, audi, bmw, chevrolet, citroen, dacia, daewoo, dodge, ferrari, fiat, ford, honda, hyundai, jaguar, jeep, kia, lada, lancia, landrover, lexus, masserati, mazda, mercedes, mitshubisi, nissan, opel, pegueot, porsche, renault, rover, saab, seat, skoda, subaru, suzuki, tata, tesla, toyota, volkswagen, volvo)
if alfa == maxLabel:
    label = "alfa"
elif audi == maxLabel:
    label = "audi"
elif bmw == maxLabel:
    label = "bmw"
elif chevrolet == maxLabel:
    label = "chevrolet"
elif citroen == maxLabel:
    label = "citroen"
elif dacia == maxLabel:
    label = "dacia"
elif daewoo == maxLabel:
    label = "daewoo"
elif dodge == maxLabel:
    label = "dodge"
elif ferrari == maxLabel:
    label = "ferrari"
elif fiat == maxLabel:
    label = "fiat"
elif ford == maxLabel:
    label = "ford"
elif honda == maxLabel:
    label = "honda"
elif hyundai == maxLabel:
    label = "hyundai"
elif jaguar == maxLabel:
    label = "jaguar"
elif jeep == maxLabel:
    label = "jeep"
elif kia == maxLabel:
    label = "kia"
elif lada == maxLabel:
    label = "lada"
elif lancia == maxLabel:
    label = "lancia"
elif landrover == maxLabel:
    label = "landrover"
elif lexus == maxLabel:
    label = "lexus"
elif masserati == maxLabel:
    label = "masserati"
elif mazda == maxLabel:
    label = "mazda"
elif mercedes == maxLabel:
    label = "mercedes"
elif mitshubisi == maxLabel:
    label = "mitshubisi"
elif nissan == maxLabel:
    label = "nissan"
elif opel == maxLabel:
    label = "opel"
elif pegueot == maxLabel:
    label = "pegueot"
elif porsche == maxLabel:
    label = "porsche"
elif renault == maxLabel:
    label = "renault"
elif rover == maxLabel:
    label = "rover"
elif saab == maxLabel:
    label = "saab"
elif seat == maxLabel:
    label = "seat"
elif skoda == maxLabel:
    label = "skoda"
elif subaru == maxLabel:
    label = "subaru"
elif suzuki == maxLabel:
    label = "suzuki"
elif tata == maxLabel:
    label = "tata"
elif tesla == maxLabel:
    label = "tesla"
elif toyota == maxLabel:
    label = "toyota"
elif volkswagen == maxLabel:
    label = "volkswagen"
elif volvo == maxLabel:
    label = "volvo"

proba = maxLabel
label = "{}: {:.2f}%".format(label, proba * 100)
# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (100, 0, 255), 2)
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

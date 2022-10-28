import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="Path to the image to predict on")
parser.add_argument("-m", "--model", type=str, help="Path to the model to use", default="model/epochs1000.hdf5")
args = parser.parse_args()

m = load_model(args.model)
img = cv2.imread(args.file)
img = np.reshape(img, (1, 80, 80, 3))
prediction = m.predict(img)

classes = ['10', '11', '12', '2', '3', '4', '5', '6', '8', '9']
idx = np.argmax(prediction)
print(classes[idx])




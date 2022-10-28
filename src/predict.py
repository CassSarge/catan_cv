print("-------------- Importing keras --------------")
print(" ")
from keras.models import load_model
print(" ")
print(" -------------- Done! --------------")
import cv2
import numpy as np
import argparse


def predictNumberFromImg(img, m):
    img = cv2.resize(img, (80,80), interpolation=cv2.INTER_AREA)
    img2 = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img = np.reshape(img2, (1, 80, 80, 3))
    prediction = m.predict(img)
    classes = ['10', '11', '12', '2', '3', '4', '5', '6', '8', '9']
    idx = np.argmax(prediction)
    return classes[idx]

def loadModel(modelpath):
    print("-------------------- Loading CNN, please wait --------------------")
    print(" ")
    m = load_model(modelpath)
    print(" ")
    print("-------------------- Done! --------------------")
    return m

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Path to the image to predict on")
    parser.add_argument("-m", "--model", type=str, help="Path to the model to use", default="model/epochs1000.hdf5")
    args = parser.parse_args()


    m = loadModel(args.model)

    img = cv2.imread(args.file)

    print(predictNumberFromImg(img, m))




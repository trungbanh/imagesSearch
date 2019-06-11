import cv2 
import pickle 
import keras 
from keras.models import load_model
import numpy as np
import os 
from ImagesProcess import ImagesProcess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def image2vector(img):
    return cv2.resize(img,(100,100)).flatten()

def catory2numeric(ara):
      return np.where(ara != 0)

def predict2path(var):
    return os.path.join('./data',os.listdir('./data')[var])
     

def image_to_feature_vector(image, size=(100, 100)):
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def main():
    img = cv2.imread("./test_choi.jpg")
    img = cv2.resize(img,(100,100))
    feature = image2vector(img)
    model = load_model('cnn.h5')
    predict = model.predict([[img]])
    var = catory2numeric(predict[0])[0][0]
    print ("var ne {0}".format(var))
    path = predict2path(var+1)

    print ("path ne {0}".format(path))
    imgs ,labels = ImagesProcess.getImages(path)

    rawImages = list()
    features = list()
    for image in imgs:
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
        rawImages.append(pixels)
        features.append(hist)
    
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)

    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.25, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        features, labels, test_size=0.25, random_state=42)

    print("[INFO] evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=7,n_jobs=4)
    model.fit(trainRI, trainRL)

    label = model.predict_proba([feature])

    myla = catory2numeric(label)

    print(myla)
    for img in myla[1] :
        if (img in range(0,8)) :
            img = str(0)+str(img+1) 
        else:
            img = str(img+1) 
        print(path+'/'+"img_000000"+str(img)+".jpg")
        img = cv2.imread(path+'/'+"img_000000"+str(img)+".jpg")
        cv2.imshow("result",img)
        cv2.waitKey(0)




if __name__ == "__main__":
    main()
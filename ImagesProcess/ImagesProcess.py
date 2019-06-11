import os 

import cv2 
import pickle 
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def getImages(path = "./data") :
    imgs = []
    labels = []
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path,image))
        img =cv2.resize(img,(100,100))
        imgs.append(img)
        labels.append(image)
    # with open('./tmp/'+folder+".pkl","wb") as f:
    #     pickle.dump(imgs,f)

    # with open('./tmp/'+folder+"labels.pkl","wb") as f:
    #     pickle.dump(labels,f)

    return imgs , labels


def loadLabels():
    path = './data/labels'
    mylabels =list()
    for data in os.listdir(path):
        with open(os.path.join(path,data),'rb') as labels :
            mylabels = pickle.load(labels)
    le = preprocessing.LabelEncoder()
    le.fit(mylabels)
    return le, mylabels 


def scale_fit():
    path = './data/dataTrain/'
    la, labels = loadLabels()
    scaler = StandardScaler()
    lis = list()
    for data in os.listdir(path):
        with open(os.path.join(path,data),'rb') as style :
            mystyle = pickle.load(style)
        for image in mystyle :
            fla = image.flatten()
            lis.append(fla)
    print(len(lis))
    lis = np.array(lis)
    
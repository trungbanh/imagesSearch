from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2 
import numpy as np
import os 
from src.ImagesProcess import ImageProcess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf


def catory2numeric(ara):
      return np.where(ara != 0)

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=100,
                                input_width=100,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def image2vector(img):
    return cv2.resize(img,(100,100)).flatten()

def predict2path(var):
    mypath = ""
    leters = var.split()
    for leter in leters:
        mypath = mypath + leter.capitalize()+"_"
    mypath = mypath[0:-1]
    return os.path.join('src/data',mypath)

def image_to_feature_vector(image, size=(100, 100)):
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label



def main():
    model_file = './src/model/output_graph.pb'
    file_name = './src/test_choi.jpg'
    input_mean=0
    input_std=255
    graph = load_graph(model_file)
    input_layer = "Placeholder"
    output_layer = "final_result"
    label_file = './src/model/output_labels.txt'
    t = read_tensor_from_image_file(
        file_name,
        input_height=299,
        input_width=299,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    real_name = labels[top_k[0]]

    path = predict2path(real_name)
    # print (path)

    imgs ,labels = ImageProcess.getImages(path)

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
    # (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    #     features, labels, test_size=0.25, random_state=42)

    # print("[INFO] evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=7,n_jobs=4)
    model.fit(trainRI, trainRL)
    img = cv2.imread(file_name)
    img = cv2.resize(img,(100,100))

    feature = image_to_feature_vector(image)
    label = model.predict_proba([feature])
    myla = catory2numeric(label)

    # print(myla)
    result = list()
    for img in myla[1] :
        if (img in range(0,9)) :
            img = str(0)+str(img) 
        result.append(path+'/'+"img_000000"+str(img)+".jpg")
        # print(path+'/'+"img_000000"+str(img)+".jpg")
        # img = cv2.imread(path+'/'+"img_000000"+str(img)+".jpg")
        # cv2.imshow("result",img)
        # cv2.waitKey(0)
    return result 
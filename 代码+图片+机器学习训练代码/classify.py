import cv2
import numpy as np


def calcSiftFeature(img, count):  # SIFT 特征点检测 count=10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(count)  # max number of SIFT points is 10
    kp, des = sift.detectAndCompute(gray, None)
    return des


def calcFeatVec(features, centers): # 计算特征向量
    wordCnt = 7
    featVec = np.zeros((1, wordCnt))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (wordCnt, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]  # index of the nearest center
        featVec[0][idx] += 1
    return featVec


def classify(img):  # 用训练好的数据做分类
    svm = cv2.ml.SVM_load("data/svm.clf")
    labels, centers = np.load("data/pos.npy")
    features = calcSiftFeature(img, 10)
    featVec = calcFeatVec(features, centers)
    case = np.float32(featVec)
    dict_svm = svm.predict(case)
    return int(dict_svm[1])

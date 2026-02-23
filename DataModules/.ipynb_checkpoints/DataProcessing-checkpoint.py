import glob
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_datasets():

    x1, y1 = [], []
    x2, y2 = [], []
    x3, y3 = [], []
    x4, y4 = [], []
    IMAGE_SIZE = (224, 224)
    
    for directory_path1 in glob.glob("/content/drive/MyDrive/Colab Notebooks/Main Project/database/datasets/training/rnfl_dm/*"):
        label1 = os.path.basename(directory_path1)
        for img_path1 in glob.glob(os.path.join(directory_path1, "*.png")):
            img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
            img1 = cv2.resize(img1, IMAGE_SIZE)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            x1.append(img1)
            y1.append(label1)
    
    for directory_path2 in glob.glob("/content/drive/MyDrive/Colab Notebooks/Main Project/database/datasets/training/rnfl_tm/*"):
        label2 = os.path.basename(directory_path1)
        for img_path2 in glob.glob(os.path.join(directory_path2, "*.png")):
            img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
            img2 = cv2.resize(img2, IMAGE_SIZE)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            x2.append(img2)
            y2.append(label2)
    
    for directory_path3 in glob.glob("/content/drive/MyDrive/Colab Notebooks/Main Project/database/datasets/training/gcipl_dm/*"):
        label3 = os.path.basename(directory_path1)
        for img_path3 in glob.glob(os.path.join(directory_path3, "*.png")):
            img3 = cv2.imread(img_path3, cv2.IMREAD_COLOR)
            img3 = cv2.resize(img3, IMAGE_SIZE)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            x3.append(img3)
            y3.append(label3)
    
    for directory_path4 in glob.glob("/content/drive/MyDrive/Colab Notebooks/Main Project/database/datasets/training/gcipl_tm/*"):
        label4 = os.path.basename(directory_path1)
        for img_path4 in glob.glob(os.path.join(directory_path4, "*.png")):
            img4 = cv2.imread(img_path4, cv2.IMREAD_COLOR)
            img4 = cv2.resize(img4, IMAGE_SIZE)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            x4.append(img4)
            y4.append(label4)
    
    
    #Data to Arrays
    x3 = np.array(x3)
    y3 = np.array(y3)
    
    x4 = np.array(x4)
    y4 = np.array(y4)
    
    x1 = np.array(x1)
    y1 = np.array(y1)
    
    x2 = np.array(x2)
    y2 = np.array(y2)
    
    
    #Encoding labels
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    
    train_labels_encoded1 = le1.fit_transform(y1)
    train_labels_encoded2 = le2.fit_transform(y2)
    train_labels_encoded3 = le3.fit_transform(y3)
    train_labels_encoded4 = le4.fit_transform(y4)
    
    #print("x1 shape:", len(x1))
    #print("y1 shape:", len(train_labels_encoded1))
    
    #Splitting the training dataset into training and internal validation(test) data
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, train_labels_encoded1, stratify = train_labels_encoded1, random_state=42, test_size = 0.25)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, train_labels_encoded2, stratify = train_labels_encoded2, random_state=42, test_size = 0.25)
    x4_train, x4_test, y4_train, y4_test = train_test_split(x4, train_labels_encoded4, stratify = train_labels_encoded4, random_state=42, test_size = 0.25)
    x3_train, x3_test, y3_train, y3_test = train_test_split(x3, train_labels_encoded3, stratify = train_labels_encoded3, random_state=42, test_size = 0.25)
    
    
    #Scaling the pixels of the images from 1 to 255 into 0 to 1 (Normalize pixel values to between 0 and 1)
    x1_train_scaled = x1_train / 255
    x1_test_scaled = x1_test / 255
    
    x2_train_scaled = x2_train / 255
    x2_test_scaled = x2_test / 255
    
    x3_train_scaled = x3_train / 255
    x3_test_scaled = x3_test / 255
    
    x4_train_scaled = x4_train / 255
    x4_test_scaled = x4_test / 255

    return x1_train_scaled, x1_test_scaled, x2_train_scaled, x2_test_scaled, x3_train_scaled, x3_test_scaled, x4_train_scaled, x4_test_scaled, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test

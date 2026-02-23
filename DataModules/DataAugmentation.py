#import importlib
#import DataModules.DataProcessing
#importlib.reload(DataModules.DataProcessing)

from .DataProcessing import load_datasets
import tensorflow as tf

def get_datasets():

    x1_train_scaled, x1_test_scaled, x2_train_scaled, x2_test_scaled, x3_train_scaled, x3_test_scaled, x4_train_scaled, x4_test_scaled, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = load_datasets()
    
    train_dataset1 = tf.data.Dataset.from_tensor_slices((x1_train_scaled, y1_train))
    train_dataset1 = train_dataset1.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)
    test_dataset1 = tf.data.Dataset.from_tensor_slices((x1_test_scaled, y1_test))
    test_dataset1 = test_dataset1.batch(8).prefetch(tf.data.AUTOTUNE)

    train_dataset2 = tf.data.Dataset.from_tensor_slices((x2_train_scaled, y2_train))
    train_dataset2 = train_dataset2.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)
    test_dataset2 = tf.data.Dataset.from_tensor_slices((x2_test_scaled, y2_test))
    test_dataset2 = test_dataset2.batch(8).prefetch(tf.data.AUTOTUNE)

    train_dataset3 = tf.data.Dataset.from_tensor_slices((x3_train_scaled, y3_train))
    train_dataset3 = train_dataset3.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)
    test_dataset3 = tf.data.Dataset.from_tensor_slices((x3_test_scaled, y3_test))
    test_dataset3 = test_dataset3.batch(8).prefetch(tf.data.AUTOTUNE)

    train_dataset4 = tf.data.Dataset.from_tensor_slices((x4_train_scaled, y4_train))
    train_dataset4 = train_dataset4.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)
    test_dataset4 = tf.data.Dataset.from_tensor_slices((x4_test_scaled, y4_test))
    test_dataset4 = test_dataset4.batch(8).prefetch(tf.data.AUTOTUNE)

    #Data Augmentation Layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])
    
    return train_dataset1, test_dataset1, train_dataset2, test_dataset2, train_dataset3, test_dataset3, train_dataset4, test_dataset4, data_augmentation
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:38:50 2021

@author: PC
"""
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self):
        self.WIDTH = 256
        self.HEIGHT = 256
        self.image_path = []
        self.mask_path = []
        
        self.load_data()
    
    def load_data(self):
        self.image_path = [os.path.join('Dataset/images/', f'{name}') for name in os.listdir('Dataset/images')]
        self.mask_path = [os.path.join('Dataset/masks/', f'{name}') for name in os.listdir('Dataset/masks')]
    
    def separate_data(self):
        train_x, valid_x = train_test_split(self.image_path, test_size=0.2, random_state=42)
        train_y, valid_y = train_test_split(self.mask_path, test_size=0.2, random_state=42)
        
        return (train_x, train_y), (valid_x, valid_y)
            
    def read_image(self, image_path):
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.resize(image, (self.WIDTH, self.HEIGHT))
        image = image/255.0
        image = image.astype(np.float32)
        return image 
    
    def read_mask(self, mask_path):
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask = cv.resize(mask, (self.WIDTH, self.HEIGHT))
        mask = mask.astype(np.int32)
        return mask
        
    def convert2TfDataset(self, x, y, batch_size=8):
        def preprocess(image_path, mask_path):
            def f(image_path, mask_path):
                image_path = image_path.decode()
                mask_path = mask_path.decode()
                image = self.read_image(image_path)
                mask = self.read_mask(mask_path)
                return image, mask
        
            image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.int32])
            mask = tf.one_hot(mask, 47, dtype=tf.int32)
            image.set_shape([self.HEIGHT, self.WIDTH, 3])
            mask.set_shape([self.HEIGHT, self.WIDTH, 47])
            return image, mask
            
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(preprocess)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(2)
        return dataset
        
if __name__ == '__main__':
    dataset = Dataset()
    (train_x, train_y), (valid_x, valid_y) = dataset.separate_data()
    print(f'Dataset: Training: {len(train_x)} - Validation: {len(valid_x)} ')
    
    train_dataset = dataset.convert2TfDataset(train_x, train_y, 8)
    for image, mask in train_dataset:
        print(f'{image.shape} - {mask.shape}')
    
    
        
        
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:09:37 2021

@author: PC
"""

import os
import numpy as np
import tensorflow as tf
from data_preprocessing import FashionDataset
import cv2 as cv
from tqdm import tqdm
import json
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

H = 256
W = 256
num_classes=47

if __name__ == '__main__':
    
    test_x = [os.path.join('imaterialist-fashion-2019-FGVC6/test/', f'{name}') for name in os.listdir("imaterialist-fashion-2019-FGVC6/test/")]
    
    df = pd.DataFrame()
    fashion = FashionDataset(df)
    label_names = fashion.labels
    
    """ Model """
    model = tf.keras.models.load_model('mobilenetv2_unet.h5')
    
    """ Saving """
    for x in tqdm(test_x[:200]):
        name = x.split('/')[-1]
        
        # Read image
        x = cv.imread(x, cv.IMREAD_COLOR)
        x = cv.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        
        # Prediction
        p = model.predict(np.expand_dims(x, axis=0))[0]
        p = np.argmax(p, axis=-1)
        
        
        text = str([label_names[i].split(', ')[0] for i in np.unique(np.reshape(p, -1)) if i != 0])
    
        p = np.expand_dims(p, axis=-1)
        p[p != 0] = 255
        p = p.astype(np.int32)
        p = np.concatenate([p,p,p], axis=2)
        
        
        x = x*255.0
        x = x.astype(np.int32)
        bitwise_and = cv.bitwise_and(p, x)
        
        cv.putText(bitwise_and, text, (0, 250), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,250,0),1)
        # cv.imshow('anh', bitwise_and)
        line = np.ones((H, 10, 3)) * 255
        
        final_image = np.concatenate([x, line, bitwise_and], axis=1)
        cv.imwrite(f'Clothing Segmentation/test/{name}', final_image)
        
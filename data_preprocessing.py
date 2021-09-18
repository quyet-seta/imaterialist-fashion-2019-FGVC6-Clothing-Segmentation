# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:04:35 2021

@author: PC
"""
import os
import numpy as np
import cv2 as cv
import pandas as pd
import json
from progressbar import ProgressBar


class FashionDataset():
    def __init__(self, df):
        self.train_path = 'imaterialist-fashion-2019-FGVC6/train/'
        self.test_path = 'imaterialist-fashion-2019-FGVC6/test/'
        self.train_csv = 'imaterialist-fashion-2019-FGVC6/train.csv'
        self.json_file = 'imaterialist-fashion-2019-FGVC6/label_descriptions.json'
        self.labels = ['Background']
        self.df = df
        self.getLabels()

    def getLabels(self):
        with open(self.json_file, mode='r') as f:
            content = json.load(f)
        
        label_name = [x['name'] for x in content['categories']]
        self.labels.extend(label_name)
        
    def makeMaskImage(self, segment_df):
        W = segment_df.at[0, 'Width']
        H = segment_df.at[0, 'Height']
        mask = np.full(H*W, 0, dtype='uint8')
        for encodedPixel, classId in zip(segment_df['EncodedPixels'].values, segment_df['ClassId'].values):
            pixelList = list(map(int, encodedPixel.split(' ')))
            for i in range(0,len(pixelList),2):
                start_idx = pixelList[i]
                step = pixelList[i+1]
                if int(classId.split('_')[0]) < 46:
                    mask[start_idx:start_idx+step] = int(classId.split('_')[0]) + 1
        
        mask_reshape = np.reshape(mask, (H, W), order='F')
        
        return mask_reshape
    
    def makeData(self):
        image = df['ImageId'].unique()[:10000]
        pbar = ProgressBar()
        
        for img in pbar(image):
            segment_df = self.df[self.df['ImageId'] == img].reset_index()
            mask = self.makeMaskImage(segment_df)
            mask_ = np.dstack((mask, mask, mask))
            cv.imwrite(f'Clothing Segmentation/Dataset/masks/{img}', mask_)
            im = cv.imread(self.train_path+img, cv.IMREAD_COLOR)
            cv.imwrite(f'Clothing Segmentation/Dataset/images/{img}', im)


if __name__ == "__main__":
    df = pd.read_csv('imaterialist-fashion-2019-FGVC6/train.csv')
    fashionData = FashionDataset(df)
    fashionData.makeData()
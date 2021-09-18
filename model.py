# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:16:57 2021

@author: PC
"""
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import config as cfg
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from data import Dataset

def _conv_block(inputs, filters):
    x = Conv2D(filters, (3,3), 1, 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3,3), 1, 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def _decoder_block(inputs, skip_connection, filters):
    x = Conv2DTranspose(filters, (2,2), 2, 'same')(inputs)
    x = Concatenate()([x, skip_connection])
    x = _conv_block(x, filters)
    return x

class MobiNetV2Unet():
    def __init__(self, verbose=True):
        self.shape = cfg.SHAPE
        self.n_classes = cfg.CLASSES
        self.build_model()
        self.dataset = Dataset()
        if verbose:
            self.model.summary()

    def build_model(self):
        """ INPUT """
        inputs = Input(shape=self.shape, name='input')
        
        """ BACKBONE MobileNetV2 """
        encoder = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
        
        """ Encoder """
        s1 = encoder.get_layer('input').output 
        s2 = encoder.get_layer('block_1_expand_relu').output
        s3 = encoder.get_layer('block_3_expand_relu').output        
        s4 = encoder.get_layer('block_6_expand_relu').output
        
        """ Bridge """
        b1 = encoder.get_layer('block_13_expand_relu').output        

        """ Decoder """
        d1 = _decoder_block(b1, s4, 512)
        d2 = _decoder_block(d1, s3, 256)
        d3 = _decoder_block(d2, s2, 128)
        d4 = _decoder_block(d3, s1, 64)
        
        """ Output """
        outputs = Conv2D(self.n_classes, (1,1), 1, 'same', activation='softmax')(d4)
        
        self.model = Model(inputs, outputs, name='MobilenetV2_Unet')
        
    def train(self):
        (train_x, train_y), (valid_x, valid_y) = self.dataset.separate_data()
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(cfg.LEARNING_RATE),
            metrics = [
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
                tf.keras.metrics.MeanIoU(num_classes=self.n_classes)
                ]
            )
        
        callbacks = [
                ModelCheckpoint(cfg.WEIGHT, verbose=1, save_best_model=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, min_lr=1e-6),
                CSVLogger(cfg.CSV_LOGGER),
                EarlyStopping(monitor="val_loss", patience=5, verbose=1)
            ]
        
        
        train_dataset = self.dataset.convert2TfDataset(train_x, train_y, batch_size=cfg.BATCH_SIZE)
        valid_dataset = self.dataset.convert2TfDataset(valid_x, valid_y, batch_size=cfg.BATCH_SIZE)
        
        train_step = len(train_x)//cfg.BATCH_SIZE
        if len(train_x) % cfg.BATCH_SIZE != 0:
            train_step += 1
            
        valid_step = len(valid_x)//cfg.BATCH_SIZE
        if len(valid_x) % cfg.BATCH_SIZE != 0:
            valid_step += 1
        
        self.model.fit(train_dataset,
                       validation_data=valid_dataset,
                       steps_per_epoch=train_step,
                       validation_steps=valid_step,
                       epochs=cfg.EPOCHS,
                       callbacks=callbacks)
        
if __name__ == "__main__":
    model = MobiNetV2Unet()
    model.train()
        
        
        
        
        
        
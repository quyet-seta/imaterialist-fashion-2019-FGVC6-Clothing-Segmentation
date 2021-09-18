# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:16:57 2021

@author: PC
"""
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPool2D, Concatenate, AveragePooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import config as cfg
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from data import Dataset

""" Atrous Spatial Pyramid Pooling """
def ASPP(inputs):
    shape = inputs.shape

    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

class DeepLabV3Plus():
    def __init__(self, verbose=True):
        self.shape = cfg.SHAPE
        self.n_classes = cfg.CLASSES
        self.build_model()
        self.dataset = Dataset()
        if verbose:
            self.model.summary()

    def build_model(self):
        """ Inputs """
        inputs = Input(self.shape)
        
        """ Pre-trained ResNet50 """
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        
        """ Pre-trained ResNet50 Output """
        image_features = base_model.get_layer('conv4_block6_out').output
        x_a = ASPP(image_features)
        x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)
        
        """ Get low-level features """
        x_b = base_model.get_layer('conv2_block2_out').output
        x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
        x_b = BatchNormalization()(x_b)
        x_b = Activation('relu')(x_b)
        
        x = Concatenate()([x_a, x_b])
        
        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((4, 4), interpolation="bilinear")(x)
        
        """ Outputs """
        x = Conv2D(self.n_classes, (1, 1), name='output_layer')(x)
        x = Activation('softmax')(x)
        
        """ Model """
        self.model = Model(inputs=inputs, outputs=x)
        
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
    model = DeepLabV3Plus()
   # model.train()
        
        
        
        
        
        
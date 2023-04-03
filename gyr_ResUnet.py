
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from random import shuffle
import tensorflow as tf

import datetime
import tensorflow.keras.optimizers as keras_optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv3D, UpSampling3D, BatchNormalization
from tensorflow.keras.layers import Activation, add, concatenate, MaxPooling3D, Dropout
import keras.backend as K
import tensorflow.keras.initializers as K_initializer
from sklearn.utils import shuffle

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='elu')(res_path)
    res_path = Conv3D(filters=nb_filters[0], kernel_size=(3, 3, 3), padding='same', strides=strides[0], kernel_initializer=K_initializer.he_normal(seed=None))(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='elu')(res_path)
    res_path = Conv3D(filters=nb_filters[1], kernel_size=(3, 3, 3), padding='same', strides=strides[1], kernel_initializer=K_initializer.he_normal(seed=None))(res_path)

    shortcut = Conv3D(nb_filters[1], kernel_size=(1, 1, 1), strides=strides[0], kernel_initializer=K_initializer.he_normal(seed=None))(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path

def encoder(x):
    to_decoder = []

    main_path = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), kernel_initializer=K_initializer.he_normal(seed=None))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='elu')(main_path)
    main_path = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), kernel_initializer=K_initializer.he_normal(seed=None))(main_path)
    shortcut  = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), kernel_initializer=K_initializer.he_normal(seed=None))(x)
    shortcut  = BatchNormalization()(shortcut)
    main_path = add([shortcut, main_path])

    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128, 128], [(2, 2, 2), (1, 1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256, 256], [(2, 2, 2), (1, 1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [512, 512, 512], [(2, 2, 2), (1, 1, 1)])
    to_decoder.append(main_path)
    
    #main_path = res_block(main_path, [1024, 1024, 1024], [(2, 2, 2), (1, 1, 1)])
    #to_decoder.append(main_path)

    return to_decoder

def decoder(x, from_encoder):
    #main_path = UpSampling3D(size=(2, 2, 2))(x)
    #main_path = concatenate([main_path, from_encoder[4]], axis=4)
    #main_path = res_block(main_path, [1024, 1024, 1024], [(1, 1, 1), (1, 1, 1)])
    
    main_path = UpSampling3D(size=(2, 2, 2))(x)
    main_path = concatenate([main_path, from_encoder[3]], axis=4)
    main_path = res_block(main_path, [512, 512, 512], [(1, 1, 1), (1, 1, 1)])

    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[2]], axis=4)
    main_path = res_block(main_path, [256, 256, 256], [(1, 1, 1), (1, 1, 1)])

    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=4)
    main_path = res_block(main_path, [128, 128, 128], [(1, 1, 1), (1, 1, 1)])

    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=4)
    main_path = res_block(main_path, [64, 64, 64], [(1, 1, 1), (1, 1, 1)])
    return main_path

def build_res_unet(input_shape):
    inputs_1 = Input(shape=input_shape)
    to_decoder = encoder(inputs_1)
    path = res_block(to_decoder[3], [1024, 1024, 1024], [(2, 2, 2), (1, 1, 1)])
    path = decoder(path, from_encoder=to_decoder)
    path = Dropout(0.05)(path) #Not in the original architecture
    path = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid', kernel_initializer=K_initializer.he_normal(seed=None))(path)
    return Model(inputs=inputs_1, outputs=path)

# hyper parameters
pix_size = 32
input_shape = (pix_size, pix_size, pix_size, 3)
batch_size = 128
num_epochs = 50
path_Data = "/content" # path to read the data
path = "/content/Results" # path for saving results
all_files = glob.glob(path_Data + "/*.txt")
shuffle(all_files)
Data = []
for filename in all_files:
    tmp = pd.read_csv(filename, sep="\s+", header=None)
    tmp = tmp.values
    Data.append(tmp)
Data = np.array(Data)
np.shape(Data)

Data = np.concatenate(([Data[0], Data[1]]))#, Data[2], Data[3],Data[4], Data[5], Data[6], Data[7], Data[8], Data[9], Data[10], Data[11]])) 
num_examples = Data.shape[0]
num_examples
model = build_res_unet(input_shape=input_shape)
model.summary()

x = np.empty((num_examples,32,32,32,3))
y = np.empty((num_examples,32,32,32,1))
for i in range(num_examples):

    x_ty        = Data[i,0] * np.ones((32,32,32))
    x[i,:,:,:,0]  = x_ty
    
    x_mf        = Data[i,1] * np.ones((32,32,32))
    x[i,:,:,:,1]  = x_mf
    
    x_rm        = Data[i,2] * np.ones((32,32,32))
    x[i,:,:,:,2]  = x_rm
    
    y0 =  Data[i,3:]
    y0 =  y0.reshape(32,32,32)
    y[i,:,:,:,0]  = y0
    
x, y = shuffle(x, y)

optimizer = keras_optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True)
loss_function='mean_squared_error'
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[dice_coef])

train_fraction = 0.92
train_range    = int(num_examples * train_fraction)
x_train        = x[:train_range,:,:,:,:]
y_train        = y[:train_range,:,:,:,:]
print(y_train.shape)

x_test           = x[train_range:,:,:,:,:]
y_test_gt        = y[train_range:,:,:,:,:]
print(y_test_gt.shape)

history = model.fit(x=x_train, y=y_train, validation_split=0.01,
 batch_size=batch_size, epochs=num_epochs, verbose=2, shuffle=True)

loss_history = history.history["loss"]
np_loss_history = np.array(loss_history)
np.savetxt(path+"/loss_history.txt", np_loss_history, delimiter=",")

v_loss_history = history.history["val_loss"]
np_v_loss_history = np.array(v_loss_history)
np.savetxt(path+"/v_loss_history.txt", np_v_loss_history, delimiter=",")

DC_history  = history.history["dice_coef"]
np_DC_history = np.array(DC_history)
np.savetxt(path + "/DC_history.txt", np_DC_history, delimiter=",")

val_DC_history  = history.history["val_dice_coef"]
np_val_DC_history = np.array(val_DC_history)
np.savetxt(path + "/val_DC_history.txt", np_val_DC_history, delimiter=",")

model_evaluation = model.evaluate(x=x_test, y=y_test_gt, batch_size=batch_size)
print(model_evaluation)

y_test_pred = model.predict(x_test)

y_gt   = np.squeeze(y_test_gt)
y_gt   = y_gt.reshape(len(y_gt),pix_size*pix_size*pix_size)
df_y_gt = pd.DataFrame(data=y_gt)
df_y_gt.to_csv(path + "/y_gt_data",index=False,header=None)
y_gt_data= pd.read_csv(path + "/y_gt_data",header=None)

y_pred = np.squeeze(y_test_pred)
y_pred = y_pred.reshape(len(y_pred),pix_size*pix_size*pix_size)
df_y_pred = pd.DataFrame(data=y_pred)
df_y_pred.to_csv(path + "/y_pred_data",index=False,header=None)
y_pred_data= pd.read_csv(path + "/y_pred_data",header=None)

x_ty   = np.squeeze(x_test[:,:,:,:,0])
x_ty   = x_ty.reshape(len(y_gt),pix_size*pix_size*pix_size)
df_x_ty = pd.DataFrame(data=x_ty)
df_x_ty.to_csv(path + "/x_ty_data",index=False,header=None)
x_ty_data= pd.read_csv(path + "/x_ty_data",header=None)

x_mf   = np.squeeze(x_test[:,:,:,:,1])
x_mf   = x_mf.reshape(len(y_gt),pix_size*pix_size*pix_size)
df_x_mf = pd.DataFrame(data=x_mf)
df_x_mf.to_csv(path + "/x_mf_data",index=False,header=None)
x_mf_data= pd.read_csv(path + "/x_mf_data",header=None)

x_rm   = np.squeeze(x_test[:,:,:,:,2])
x_rm   = x_rm.reshape(len(y_gt),pix_size*pix_size*pix_size)
df_x_rm = pd.DataFrame(data=x_rm)
df_x_rm.to_csv(path + "/x_rm_data",index=False,header=None)
x_rm_data= pd.read_csv(path + "/x_rm_data",header=None)

try:
    model.save(path + "/gyr_mat.h5")
    print("Saved model to disk")
except:
    pass
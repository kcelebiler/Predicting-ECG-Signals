from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional, Flatten, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
from scipy.io import loadmat
import ecg_plot
SNOMED_scored = pd.read_csv(r"C:\Users\User\OneDrive\Masaüstü\ecg form app\SNOMED_mappings_scored.csv", sep=";")
SNOMED_unscored = pd.read_csv(r"C:\Users\User\OneDrive\Masaüstü\ecg form app\SNOMED_mappings_unscored.csv", sep=";")
snomed_class_names=["Pacing Rhythm", "Prolonged QT Interval","Atrial Fibrillation","Atrial Flutter",
                 "Left Bundle Branch Block","Q Wave Abnormal","T Wave Abnormal","Prolonged PR Interval","Ventricular Premature Beats",
"Low QRS Voltages","1st Degree AV Block","Premature Atrial Contraction","Left Axis Deviation",
"Sinus Bradycardia","Bradycardia","Sinus Rhythm","Sinus Tachycardia","Premature Ventricular Contractions",
"Sinus Arrhythmia","Left Anterior Fascicular Block","Right Axis Deviation","Right Bundle Branch Block","T Wave Inversion",
"Supraventricular Premature Beats","Nonspecific Intraventricular Conduction Disorder","Incomplete Right Bundle Branch Block",
"Complete Right Bundle Branch Block"]

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

root=tk.Tk()
root.geometry("270x230")
root.title('Predict ECG Signals') 
def get_actual_disease():
    global disease_scored
    global disease_unscored
    
    with open(ecg_hea_name, 'r') as the_file:
        all_data = [line.strip() for line in the_file.readlines()]
        data = all_data[8:]
    
    snomed_number=int(data[7][5:14])
    value_unscored=SNOMED_unscored["Dx"][SNOMED_unscored["SNOMED CT Code"]==snomed_number].values
    value_scored=SNOMED_scored["Dx"][SNOMED_scored["SNOMED CT Code"]==snomed_number].values
    try: 
        disease_unscored=value_unscored[0]
        actual.set("Actual: "+disease_unscored)
    except:
        disease_unscored=""
    
    try: 
        disease_scored=value_scored[0]
        actual.set("Actual: "+disease_scored)
    except:
        disease_scored=""

def open_file():
    global ecg_mat_name
    global ecg_hea_name
    filetypes = (
        ('All files', '*.*')
    )
    file = fd.askopenfile()
    ecg_mat_name=file.name
    ecg_hea_name=file.name[:-3]+"hea"
    ecg_data = load_challenge_data(ecg_mat_name)
    ecg_plot.plot(ecg_data[0]/1000, sample_rate=500, title='')
    ecg_plot.show()


def identity_block(X, f, filters):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv1D(filters = F1, kernel_size = 1, activation='relu', strides = 1, padding = 'valid')(X)
    X = BatchNormalization()(X)
    X = Conv1D(filters = F2, kernel_size = f, activation='relu', strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(filters = F3, kernel_size = 1, activation='relu', strides = 1, padding = 'valid')(X)
    X = BatchNormalization()(X)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, s = 2):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv1D(F1, 1, activation='relu', strides = s)(X)
    X = BatchNormalization()(X)
    X = Conv1D(F2, f, activation='relu', strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Conv1D(F3, 1, strides = 1, activation='relu')(X)
    X = BatchNormalization()(X)
    X_shortcut = Conv1D(F3, 1, strides = s)(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X

def ResNet50(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding1D(3)(X_input)
    X = Conv1D(64, 7, strides = 2, activation='relu')(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(pool_size=2, strides=2, padding='same')(X)
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    X = convolutional_block(X, f = 3, filters = [128,128,512], s = 2)
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])
    X = MaxPool1D(pool_size=2, strides=2, padding='same')(X)
    X = GlobalAveragePooling1D()(X)
    X = Dense(27,activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model



def load_models():
    def inception_block(prev_layer):
        conv1=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
        conv1=BatchNormalization()(conv1)
        conv1=Activation('relu')(conv1)
        conv3=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
        conv3=BatchNormalization()(conv3)
        conv3=Activation('relu')(conv3)
        conv3=Conv1D(filters = 64, kernel_size = 3, padding = 'same')(conv3)
        conv3=BatchNormalization()(conv3)
        conv3=Activation('relu')(conv3)
        conv5=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
        conv5=BatchNormalization()(conv5)
        conv5=Activation('relu')(conv5)
        conv5=Conv1D(filters = 64, kernel_size = 5, padding = 'same')(conv5)
        conv5=BatchNormalization()(conv5)
        conv5=Activation('relu')(conv5)
        pool= MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
        convmax=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(pool)
        convmax=BatchNormalization()(convmax)
        convmax=Activation('relu')(convmax)
        layer_out = concatenate([conv1, conv3, conv5, convmax], axis=1)
        return layer_out

    def inception_model(input_shape):
        X_input=Input(input_shape)
        X = Conv1D(filters = 64, kernel_size = 1, padding = 'same')(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv1D(filters = 64, kernel_size = 1, padding = 'same')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = inception_block(X)
        X = MaxPool1D(pool_size=2, strides=4, padding='same')(X)
        X = inception_block(X)
        X = MaxPool1D(pool_size=2, strides=4, padding='same')(X)
        X = inception_block(X)
        X = MaxPool1D(pool_size=2, strides=4, padding='same')(X)
        X = inception_block(X)
        X = MaxPool1D(pool_size=2, strides=4, padding='same')(X)
        X = inception_block(X)
        X = MaxPool1D(pool_size=2, strides=4, padding='same')(X)
        X = GlobalAveragePooling1D()(X)
        X = Dense(64,activation='relu')(X)
        X = Dense(64,activation='relu')(X)
        X = Dense(27,activation='softmax')(X)
        model = Model(inputs = X_input, outputs = X, name='Inception')
        return model
    open_file()
    get_actual_disease()
    global ann_prediction
    global lenet5_prediction
    global alexnet_prediction
    global vgg16_prediction
    global resnet50_prediction
    global inception_prediction
    global lstm_prediction
    
    ann_model = Sequential()
    ann_model.add(Dense(50, activation='relu', input_shape=(5000,12)))
    ann_model.add(Dense(50, activation='relu'))
    ann_model.add(GlobalAveragePooling1D())
    ann_model.add(Dense(27, activation='softmax'))
    ann_model.load_weights("C://Users//User//OneDrive//Masaüstü//ecg form app//ecg models//ann_model_weights.best.hdf5")
    yhat=ann_model.predict(x=loadmat(ecg_mat_name)['val'].reshape(1,loadmat(ecg_mat_name)['val'].shape[1],loadmat(ecg_mat_name)['val'].shape[0]))
    ann_prediction = snomed_class_names[np.argmax(yhat)]
    ann.set("ANN Prediction: "+ann_prediction)
    
    lenet_5_model=Sequential()
    lenet_5_model.add(Conv1D(filters=64, kernel_size=5, padding='same', input_shape=(5000,12)))
    lenet_5_model.add(BatchNormalization())
    lenet_5_model.add(Activation('relu'))
    lenet_5_model.add(Conv1D(filters=64, kernel_size=3, padding='same',))
    lenet_5_model.add(BatchNormalization())
    lenet_5_model.add(Activation('relu'))
    lenet_5_model.add(GlobalAveragePooling1D())
    lenet_5_model.add(Dense(64, activation='relu'))
    lenet_5_model.add(Dense(64, activation='relu'))
    lenet_5_model.add(Dense(27, activation = 'softmax'))
    lenet_5_model.load_weights("C://Users//User//OneDrive//Masaüstü//ecg form app//ecg models//lenet5_model_weights.best.hdf5")
    yhat=lenet_5_model.predict(x=loadmat(ecg_mat_name)['val'].reshape(1,loadmat(ecg_mat_name)['val'].shape[1],loadmat(ecg_mat_name)['val'].shape[0]))
    lenet5_prediction = snomed_class_names[np.argmax(yhat)]
    lenet5.set("LeNet-5 Prediction: "+lenet5_prediction)
    
    alexNet_model=Sequential()
    alexNet_model.add(Conv1D(filters=96, kernel_size=11, padding='same', input_shape=(5000,12)))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(Conv1D(filters=384, padding='same', activation='relu', kernel_size=3))
    alexNet_model.add(Conv1D(filters=384, activation='relu', kernel_size=3))
    alexNet_model.add(Conv1D(filters=256, kernel_size=3))
    alexNet_model.add(BatchNormalization())
    alexNet_model.add(Activation('relu'))
    alexNet_model.add(GlobalAveragePooling1D())
    alexNet_model.add(Dense(64, activation='relu'))
    alexNet_model.add(Dropout(0.25))
    alexNet_model.add(Dense(64, activation='relu'))
    alexNet_model.add(Dropout(0.25))
    alexNet_model.add(Dense(27, activation='softmax'))
    alexNet_model.load_weights("C://Users//User//OneDrive//Masaüstü//ecg form app//ecg models//alexnet_model_weights.best.hdf5")
    yhat=alexNet_model.predict(x=loadmat(ecg_mat_name)['val'].reshape(1,loadmat(ecg_mat_name)['val'].shape[1],loadmat(ecg_mat_name)['val'].shape[0]))
    alexnet_prediction=snomed_class_names[np.argmax(yhat)]
    alexnet.set("AlexNet Prediction: "+alexnet_prediction)
    
    vgg_16_model=Sequential()
    vgg_16_model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(5000,12)))
    vgg_16_model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=1, activation='relu', padding='same'))
    vgg_16_model.add(Conv1D(filters=512, kernel_size=1, activation='relu', padding='same'))
    vgg_16_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
    vgg_16_model.add(BatchNormalization())
    vgg_16_model.add(GlobalAveragePooling1D())
    vgg_16_model.add(Dense(256, activation='relu'))
    vgg_16_model.add(Dropout(0.25))
    vgg_16_model.add(Dense(128, activation='relu'))
    vgg_16_model.add(Dropout(0.25))
    vgg_16_model.add(Dense(27, activation='softmax'))
    vgg_16_model.load_weights("C://Users//User//OneDrive//Masaüstü//ecg form app//ecg models//vgg16_model_weights.best.hdf5")
    yhat=vgg_16_model.predict(x=loadmat(ecg_mat_name)['val'].reshape(1,loadmat(ecg_mat_name)['val'].shape[1],loadmat(ecg_mat_name)['val'].shape[0]))
    vgg16_prediction=snomed_class_names[np.argmax(yhat)]
    vgg16.set("VGG-16 Prediction: "+vgg16_prediction)
    
    resNet50_model = ResNet50(input_shape = (5000,12))
    resNet50_model.load_weights("C://Users//User//OneDrive//Masaüstü//ecg form app//ecg models//resnet50_model_weights.best.hdf5")
    yhat=resNet50_model.predict(x=loadmat(ecg_mat_name)['val'].reshape(1,loadmat(ecg_mat_name)['val'].shape[1],loadmat(ecg_mat_name)['val'].shape[0]))
    resnet50_prediction = snomed_class_names[np.argmax(yhat)]
    resnet50.set("ResNet50 Prediction: "+resnet50_prediction)

    inception_model = inception_model(input_shape = (5000,12))
    inception_model.load_weights("C://Users//User//OneDrive//Masaüstü//ecg form app//ecg models//inception_model2_weights.best.hdf5")
    yhat=inception_model.predict(x=loadmat(ecg_mat_name)['val'].reshape(1,loadmat(ecg_mat_name)['val'].shape[1],loadmat(ecg_mat_name)['val'].shape[0]))
    inception_prediction = snomed_class_names[np.argmax(yhat)]
    inception.set("Inception Prediction: "+inception_prediction)

    lstm_model = Sequential()
    lstm_model.add(LSTM(64, return_sequences=True, input_shape=(5000,12)))
    lstm_model.add(LSTM(64, return_sequences=True))
    lstm_model.add(LSTM(32, return_sequences=True))
    lstm_model.add(GlobalAveragePooling1D())
    lstm_model.add(Dense(32, activation = 'relu'))
    lstm_model.add(Dense(27, activation = 'softmax'))
    lstm_model.load_weights("C://Users//User//OneDrive//Masaüstü//ecg form app//ecg models//lstm_model_weights.best.hdf5")
    yhat=lstm_model.predict(x=loadmat(ecg_mat_name)['val'].reshape(1,loadmat(ecg_mat_name)['val'].shape[1],loadmat(ecg_mat_name)['val'].shape[0]))
    lstm_prediction = snomed_class_names[np.argmax(yhat)]
    lstm.set("LSTM Prediction: "+lstm_prediction)
        
open_button = ttk.Button(
    root,
    text='Open an ECG File',
    command=load_models
)

open_button.grid(column=0, row=0, sticky='w', padx=90, pady=10)

actual=tk.StringVar()
actual.set("Actual: ")
actual_label = tk.Label(root, textvariable=actual)
actual_label.grid(column=0,row=1)
ann=tk.StringVar()
ann.set("ANN Prediction: ")
ann_label = tk.Label(root, textvariable=ann)
ann_label.grid(column=0,row=2)
lenet5=tk.StringVar()
lenet5.set("LeNet-5 Prediction: ")
lenet5_label = tk.Label(root, textvariable=lenet5)
lenet5_label.grid(column=0,row=3)
alexnet=tk.StringVar()
alexnet.set("AlexNet Prediction: ")
alexnet_label = tk.Label(root, textvariable=alexnet)
alexnet_label.grid(column=0,row=4)
vgg16=tk.StringVar()
vgg16.set("VGG-16 Prediction: ")
vgg16_label = tk.Label(root, textvariable=vgg16)
vgg16_label.grid(column=0,row=5)
resnet50=tk.StringVar()
resnet50.set("ResNet50 Prediction: ")
resnet50_label = tk.Label(root, textvariable=resnet50)
resnet50_label.grid(column=0,row=6)
inception=tk.StringVar()
inception.set("Inception Prediction: ")
inception_label = tk.Label(root, textvariable=inception)
inception_label.grid(column=0,row=7)
lstm=tk.StringVar()
lstm.set("LSTM Prediction: ")
lstm_label = tk.Label(root, textvariable=lstm)
lstm_label.grid(column=0,row=8)

if __name__ == "__main__":
    root.mainloop()
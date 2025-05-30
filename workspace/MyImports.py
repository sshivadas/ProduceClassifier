# Import python libraries for data processing and visualisation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import  warnings
warnings.filterwarnings("ignore")
#libraries imported to handle image data
import os, random, glob
from pathlib import Path

# Tensorflow import for model creation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, ReLU, Softmax, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Sklear imports for model evaluation
import sklearn.metrics as metrics


#Streamlit
import streamlit as st
#Pillow
from PIL import Image

#Global Variables
# Specifying the image size to resize all images
image_size = (256, 256)

#Class names with index
# index           0                1        2         3
class_names = ['indian market', 'onion', 'potato', 'tomato']

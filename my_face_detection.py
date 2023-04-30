import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from matplotlib import pyplot as plt

from pathlib import Path
import numba


img_directory_path = '/media/alpha/Samsung_T5/deepLearning/my_face/'
img_directory_path_object = Path('/media/alpha/Samsung_T5/deepLearning/my_face/')
imgs = list(img_directory_path_object.glob('*.jpg'))






def create_model():
    pass
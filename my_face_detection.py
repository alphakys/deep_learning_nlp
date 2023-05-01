import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from matplotlib import pyplot as plt

from pathlib import Path
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

img_directory_path = '/media/alpha/Samsung_T5/deepLearning/my_face/'
img_directory_path_object = Path('/media/alpha/Samsung_T5/deepLearning/my_face/')
imgs_path = list(img_directory_path_object.glob('*.jpg'))

imgs_link = [str(img) for img in imgs_path]
cnt = len(imgs_link)

# cv2에서 resize할 때는 행렬의 순서를 바꿔야한다.
input_shape = (1996, 1592)
cv_input_shape = (1592, 1996)


def img_gen():
    for img in imgs_link:
        img_origin = cv2.imread(img)
        img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_origin, dsize=cv_input_shape, interpolation=cv2.INTER_LINEAR)
        yield resized_img


# train_imges[i] = resized_img
train_imges = list(img_gen())
train_imges = np.array(train_imges)

train_images, test_images = train_test_split(train_imges, shuffle=True)



def create_model():
    pass

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from matplotlib import pyplot as plt

from pathlib import Path
import numba
import cv2

img_directory_path = '/media/alpha/Samsung_T5/deepLearning/my_face/'
img_directory_path_object = Path('/media/alpha/Samsung_T5/deepLearning/my_face/')
imgs_path = list(img_directory_path_object.glob('*.jpg'))

imgs_link = [str(img) for img in imgs_path]

for img in imgs_link:
    cv2.imre

# img0 = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
# img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
# plt.imshow(img0)
# plt.show()
#
# img_resized = cv2.resize(img0, dsize=(500,500))

img_ndarray = []
for img in imgs_link:
    img_origin = cv2.imread(img)
    resized_img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    img_ndarray.append(resized_img)

def create_model():
    pass
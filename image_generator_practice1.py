from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

# url = 'https://www.sciencenews.org/wp-content/uploads/2020/03/033120_HT_covid-cat_feat-1028x579.jpg'
#
# os.system('curl ' + url + ' > 033120_HT_covid-cat_feat-1028x579.jpg')
image = cv2.cvtColor(cv2.imread('033120_HT_covid-cat_feat-1028x579.jpg'), cv2.COLOR_BGR2RGB)

img_directory_path = '/media/alpha/Samsung_T5/deepLearning/my_face/'
img_directory_path_object = Path('/media/alpha/Samsung_T5/deepLearning/my_face/')
imgs_path = sorted(img_directory_path_object.glob('*.jpg'))

imgs_link = [str(img) for img in imgs_path]


def imgshow(img):
    plt.imshow(img)
    plt.show()


def img_gen(imgs_link_list):
    for img in imgs_link_list[:10]:
        img_origin = cv2.imread(img)
        img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
        # resized_img = cv2.resize(img_origin, dsize=input_shape, interpolation=cv2.INTER_LINEAR)

        yield img_origin


def show_aug_image_batch(image_batch):
    # ImageDataGenerator는 여러개의 image를 입력으로 받음. 따라서 3차원이 아니라 batch를 포함한 4차원 array를 입력해야 한다.

    # 만약 4차원이 아니라면 이 과정을 통해서 4차원으로 만들어줘야 한다.
    # image_batch = np.expand_dims(image, axis=0)

    # image generator 적용. fit후 flow()에 image batch를 넣어줘야 함
    data_generator.fit(image_batch)
    data_gen_iter = data_generator.flow(image_batch)
    aug_image_batch = next(data_gen_iter)
    aug_image = np.squeeze(aug_image_batch)
    aug_image = aug_image.astype('int')

    N_IMAGES = 4
    fig, axes = plt.subplots(nrows=1, ncols=N_IMAGES, figsize=(22, 8))

    for i in range(N_IMAGES):
        axes[i].imshow(aug_image[i])
        axes[i].axis('off')
        fig.show()


img_batch = np.array(list(img_gen(imgs_link)), dtype=np.uint8)

# data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# show_aug_image_batch(img_batch)

# data_generator = ImageDataGenerator(width_shift_range=0.5)
# show_aug_image_batch(img_batch)

# 위쪽 또는 아래쪽 이미지 이동을 주어진 range만큼 이동한다.
# data_generator = ImageDataGenerator(height_shift_range=0.5)
# show_aug_image_batch(img_batch)

# 빈공간은 가장 가까운 곳의 픽셀값으로 채움
# data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='nearest')
# show_aug_image_batch(img_batch)

# 빈공간 만큼의 영역을 근처 공간으로 채움
# data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='reflect')
# show_aug_image_batch(img_batch)

# 빈공간 만큼의 영역을 잘려나간 이미지로 채움
# data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='wrap')
# show_aug_image_batch(img_batch)

# 특정 픽셀값으로 채움. 이때 특정 픽셀값은 cval 값으로 채움
data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='constant', cval=100)
show_aug_image_batch(img_batch)



img_batch = np.array(list(img_gen(imgs_link)), dtype=np.uint8)

# data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# show_aug_image_batch(img_batch)

# data_generator = ImageDataGenerator(width_shift_range=0.5)
# show_aug_image_batch(img_batch)

# 위쪽 또는 아래쪽 이미지 이동을 주어진 range만큼 이동한다.
# data_generator = ImageDataGenerator(height_shift_range=0.5)
# show_aug_image_batch(img_batch)

# 빈공간은 가장 가까운 곳의 픽셀값으로 채움
# data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='nearest')
# show_aug_image_batch(img_batch)

# 빈공간 만큼의 영역을 근처 공간으로 채움
# data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='reflect')
# show_aug_image_batch(img_batch)

# 빈공간 만큼의 영역을 잘려나간 이미지로 채움
# data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='wrap')
# show_aug_image_batch(img_batch)

# 특정 픽셀값으로 채움. 이때 특정 픽셀값은 cval 값으로 채움
data_generator = ImageDataGenerator(height_shift_range=0.5, fill_mode='constant', cval=100)
show_aug_image_batch(img_batch)
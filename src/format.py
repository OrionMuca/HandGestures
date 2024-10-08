# import os
# import numpy as np
# from os import listdir
# from matplotlib.pyplot import imread
# from skimage.transform import resize
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
#
# # Settings:
# img_size = 64
# grayscale_images = True
# num_class = 10
# test_size = 0.2
#
# def get_img(data_path):
#     # Getting image array from path:
#     img = imread(data_path)
#     img = resize(img, (img_size, img_size, 1 if grayscale_images else 3))
#     return img
#
# def get_dataset(dataset_path='Dataset'):
#     # Getting all data from data path:
#     try:
#         X = np.load('../npy_dataset/X.npy')
#         Y = np.load('../npy_dataset/Y.npy')
#     except:
#         labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Ensuring label order
#         X = []
#         Y = []
#         for i, label in enumerate(labels):
#             datas_path = os.path.join(dataset_path, label)
#             for data in listdir(datas_path):
#                 img = get_img(os.path.join(datas_path, data))
#                 X.append(img)
#                 Y.append(int(label))  # Ensure labels are integer type
#         # Create dataset:
#         X = np.array(X).astype('float32') / 255.  # Normalizing the images
#         Y = np.array(Y).astype('float32')
#         Y = to_categorical(Y, num_class)
#         if not os.path.exists('../npy_dataset/'):
#             os.makedirs('../npy_dataset/')
#         np.save('../npy_dataset/X.npy', X)
#         np.save('../npy_dataset/Y.npy', Y)
#     X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
#     return X, X_test, Y, Y_test
#
# if __name__ == '__main__':
#     X, X_test, Y, Y_test = get_dataset()
#     print(X.shape)
#     print(X_test.shape)
#     print(Y.shape)
#     print(Y_test.shape)

import os
import numpy as np
from os import listdir
from matplotlib.pyplot import imread
from skimage.transform import resize
from skimage import exposure, util
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Settings:
img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.2

def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path)
    img = resize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img


def is_high_quality(img):
    # Check if the image is of high quality based on contrast
    contrast = exposure.is_low_contrast(img, fraction_threshold=0.35)
    return not contrast

def augment_image(img):
    # Apply random transformations to the image
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    img = img.reshape((1, img_size, img_size, 1 if grayscale_images else 3))
    it = datagen.flow(img, batch_size=1)
    return it.next()[0]

def add_noise(img):
    # Add random noise to the image
    return util.random_noise(img)

def adjust_contrast(img):
    # Adjust the contrast of the image
    return exposure.adjust_gamma(img, gamma=0.4, gain=0.9)

def equalize_histogram(img):
    # Apply histogram equalization
    return exposure.equalize_hist(img)

def get_dataset(dataset_path='Dataset'):
    # Getting all data from data path:
    try:
        X = np.load('../npy_dataset/X.npy')
        Y = np.load('../npy_dataset/Y.npy')
    except:
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Ensuring label order
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = os.path.join(dataset_path, label)
            for data in listdir(datas_path):
                img = get_img(os.path.join(datas_path, data))
                if is_high_quality(img):
                    X.append(img)
                    Y.append(int(label))  # Ensure labels are integer type
                    # Apply augmentations
                    X.append(augment_image(img))
                    X.append(add_noise(img))
                    X.append(adjust_contrast(img))
                    X.append(equalize_histogram(img))
                    Y.extend([int(label)] * 4)  # Add labels for augmented images
        # Create dataset:
        X = np.array(X).astype('float32') / 255.  # Normalizing the images
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('../npy_dataset/'):
            os.makedirs('../npy_dataset/')
        np.save('../npy_dataset/X.npy', X)
        np.save('../npy_dataset/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X, X_test, Y, Y_test

if __name__ == '__main__':
    X, X_test, Y, Y_test = get_dataset()
    print(X.shape)
    print(X_test.shape)
    print(Y.shape)
    print(Y_test.shape)

# MMAI 5500 Assignment 3
# Yunyi Lian
# 219961846

import os
import cv2
from PIL import Image
from glob import glob
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def convert_video_to_images(img_folder, filename='assignment3_video.avi'):
    """
    Converts the video file (assignment3_video.avi) to JPEG images.
    Once the video has been converted to images, then this function doesn't
    need to be run again.
    
    Arguments
    ---------
    filename : (string) file name (absolute or relative path) of video file.
    img_folder : (string) folder where the video frames will be
    stored as JPEG images.
    """
    # Make the img_folder if it doesn't exist.'
    try:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    except OSError:
        print('Error')
        
    # Make sure that the abscense/prescence of path
    # separator doesn't throw an error.
    img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
    # Instantiate the video object.
    video = cv2.VideoCapture(filename)
    
    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")
    
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            im_fname = f'{img_folder}frame{i:0>4}.jpg'
            print('Captured...', im_fname)
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break

    video.release()
    cv2.destroyAllWindows()

    if i:
        print(f'Video converted\n{i} images written to {img_folder}')

def load_images(img_dir, im_width=60, im_height=44):
    """
    Reads, resizes and normalizes the extracted image frames from a folder.
    The images are returned both as a Numpy array of flattened images (i.e. the images with Arguments
    ---------
    img_dir : (string) the directory where the images are stored.
    im_width : (int) The desired width of the image.
    The default value works well.
    im_height : (int) The desired height of the image.
    The default value works well.
    Returns
    X : (numpy.array) An array of the flattened images.
    images : (list) A list of the resized images.
    """
    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()
    
    for fname in fnames:
        im = Image.open(fname)
        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))
        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)
        # Close the PIL image once converted and stored.
        im.close()
        
    # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))
    
    return X, images

input_img = keras.Input(shape=(44, 60, 3))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

tf.random.set_seed(0)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

convert_video_to_images('img_folder')
X, images = load_images('img_folder')

# The first 23 seconds are normal frames.
# To make the training set not contain frames with the anomalous object, the first 23*30=690 frames can be used.
# Roughly the first 65% of the whole data
X_train, X_test, _, _ = train_test_split(X, images, train_size=0.65, shuffle=False, random_state=1)

X_train = np.reshape(X_train, (len(X_train), 44, 60, 3))
X_test = np.reshape(X_test, (len(X_test), 44, 60, 3))

autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(X_test, X_test))

losses = []
for frame in images:
    frame = frame.reshape((1, 44, 60, 3))
    loss = autoencoder.evaluate(frame, frame, verbose=0)
    losses.append(loss)

# Anomaly object starts roughly from 690th frame, so plot the losses started from index 690
# Determined the threshold to be 0.516
plt.figure()
plt.plot(losses[690:])
plt.axhline(0.516, color='red', ls='-')
plt.show()

def predict(frame):
    """
    Argument
    --------
    frame : Video frame with shape == (44, 60, 3) and dtype == float.
    
    Return
    anomaly : A boolean indicating whether the frame is an anomaly or not.
    ------
    """
    frame = frame.reshape((1, 44, 60, 3))
    loss = autoencoder.evaluate(frame, frame, verbose=0)
    anomaly = loss > 0.516
    return anomaly

# Prediction of a frame containing anomalous object: True
# print(predict(images[730]))
# plt.figure()
# plt.imshow(images[730])
# plt.show()

# Predction of normal frame: False
# print(predict(images[200]))
# plt.figure()
# plt.imshow(images[200])
# plt.show()

autoencoder.save('model')

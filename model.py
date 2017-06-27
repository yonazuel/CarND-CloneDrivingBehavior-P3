
import numpy as np
import tensorflow as tf
import cv2
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
import json


# I store all the images paths in the 3 lists center_paths, left_paths, right_paths    
# I also store the associated steering angles in the list steering_angles
def load_paths():

    center_paths = []
    left_paths = []
    right_paths = []
    steering_angles = []

    with open('driving_log.csv', mode='r') as f:
        next(f) # The first line contains the headers
        for line in csv.reader(f):
            center_paths.append(line[0])
            left_paths.append(line[1][1:])
            right_paths.append(line[2][1:])
            steering_angles.append(float(line[3][1:]))

    return center_paths, left_paths, right_paths, steering_angles

# Since openCV loads the data in BGR instead of RGB, I convert them to RGB
def convert_to_rgb(image):
    b,g,r = cv2.split(image)
    return cv2.merge([r,g,b])

# I also want to resize my images to 64x64 (from 320x160)
def resize_image(image):
    return cv2.resize(image,(64,64))

# I can finally load and preprocess an image
def load_image(path):
    img = cv2.imread(path)
    img = convert_to_rgb(img)
    img = resize_image(img)
    return img

# I take 15% of the data out for my test set.
# Since the driving module uses only the center image,
# my test set also will only contain center images.
# But to avoid overfitting, I don't want the training set to contain 
# the left or right image of an image containted in the test set.
#
# Moreover, I am also going to set aside 15% of the data from my validation set.
# I do the test/validation split now 
# (before I enriched the data by creating new training examples)
# because I want my training set and my validation set to be totally exclusive:
# the same situation cannot produce a center image in the validation set 
# and a right or left image in the training set.
#
# Finally, in order to have a more normal distribution of steering angles 
# I will use the right and left images only if the steering angle is <> 0.
def split_train_val_test(center_paths, left_paths, right_paths, labels, shift_steer=0.2, proportion=0.15, batch_size=128):
    nb_examples = len(labels)
    nb_test_examples = int(nb_examples * proportion)

    train_paths = []
    test_paths = []
    validation_paths = []
    train_labels = []
    test_labels = []
    validation_labels = []

    test_validation_indices = random.sample(range(nb_examples), 2*nb_test_examples)
    test_indices = test_validation_indices[:nb_test_examples]
    validation_indices = test_validation_indices[nb_test_examples:]

    # For the test set, I only use the center image
    # The left and right images for these indices will never be used
    for i in test_indices:
        test_paths.append(center_paths[i])
        test_labels.append(labels[i])
    test_paths, test_labels = shuffle(test_paths, test_labels)

    # For the validation set, I use the center, left and right images
    # For the left image, I increase the steering angle by 0.2
    # For the right image, I decrease the steering angle by 0.2
    for i in validation_indices:
        l = labels[i]
        validation_paths.append(center_paths[i])
        validation_labels.append(l)
        if abs(l)>0.001:
            validation_paths.append(left_paths[i])
            validation_labels.append(l+shift_steer)
            validation_paths.append(right_paths[i])
            validation_labels.append(l-shift_steer)

    # I want my set to have a number of images that is a multiple of the batch_size 128
    nb_validation = len(validation_labels)
    nb_validation_new = int(nb_validation//batch_size) * batch_size
    validation_paths = validation_paths[:nb_validation_new]
    validation_labels = validation_labels[:nb_validation_new]
    validation_paths, validation_labels = shuffle(validation_paths, validation_labels)

    # I do the exact same thing for the training set
    for i in range(nb_examples):
        if not i in test_validation_indices:
            l = labels[i]
            train_paths.append(center_paths[i])
            train_labels.append(l)
            if abs(l)>0.001:
                train_paths.append(left_paths[i])
                train_labels.append(l+shift_steer)
                train_paths.append(right_paths[i])
                train_labels.append(l-shift_steer)

    # I want my set to have a number of images that is a multiple of the batch_size 128
    nb_train = len(train_labels)
    nb_train_new = int(nb_train//batch_size) * batch_size
    train_paths = train_paths[:nb_train_new]
    train_labels = train_labels[:nb_train_new]
    train_paths, train_labels = shuffle(train_paths, train_labels)

    return train_paths, train_labels, validation_paths, validation_labels, test_paths, test_labels

# Now that I have set aside a part of the data for testing,
# I am going to augment the data to have more points to train on.
# The first part of my data augmentation is going to be to flip the image
# and associate it with the opposite of the original steering angle.
# This should help by correcting the bias towards turning left due to the fact
# there are mainly left turns on track 1.
def reverse_image(image):
    return cv2.flip(image,1)

# I am also going to translate the images left and right and 
# modify the corresponding steering angle.
# This should help by correcting the bias towards driving straight due to the fact
# most of track 1 is a straight line. This is like creating some new recovery data.
def translate(image, steering_angle, shift_per_pix=0.05, pixel_range=15):
    nb_rows = image.shape[0]
    nb_cols = image.shape[1]
    pixel_shift = random.randint(-pixel_range,pixel_range)
    steering_shift = pixel_shift * shift_per_pix
    translation_matrix = np.float32([[1,0,pixel_shift],[0,1,0]])
    img = cv2.warpAffine(image,translation_matrix,(nb_cols,nb_rows))
    return img, steering_angle+steering_shift

# Because of storage limitation, I am going to use a generator.
# This generator will yield batches of images.
def generate_batch(paths, labels, multiplier=4, shift_per_pix=0.05, batch_size=128):
    batch_images = []
    batch_labels = []
    indices = shuffle(range(len(labels)))

    while len(indices)>=0:
        if len(batch_labels)<batch_size:
            # I load an image
            i = indices.pop()
            image = load_image(paths[i])
            label = labels[i]
            
            # I reverse it and associate the opposite steering angle
            image_r = reverse_image(image)
            label_r = - 1. * label

            batch_images.append(image)
            batch_labels.append(label)

            batch_images.append(image_r)
            batch_labels.append(label_r)

            for j in range(multiplier-1):
                # I translate the original image and shift the steering angle
                img, lab = translate(image, label, shift_per_pix)
                batch_images.append(img)
                batch_labels.append(lab)

                # I translate the reversed image and shift the steering angle
                img, lab = translate(image_r, label_r, shift_per_pix)
                batch_images.append(img)
                batch_labels.append(lab)

        else:
            gen_images = np.array(batch_images)
            gen_labels = np.array(batch_labels)
            batch_images = []
            batch_labels = []
            indices = shuffle(range(len(labels)))
            
            yield gen_images, gen_labels

def get_model():
    # I define my model
    model = Sequential()
    
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(64,64,3)))
    
    model.add(Convolution2D(3,1,1))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
        
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dropout(.5))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dropout(.5))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


def main(_):
    # Parameters
    shift_steer = 0.2
    multiplier = 4
    shift_per_pix = 0.05
    batch_size = 128

    # First I load the data
    center_paths, left_paths, right_paths, steering_angles = load_paths()
            
    train_paths, train_labels, validation_paths, validation_labels, test_paths, test_labels = \
        split_train_val_test(center_paths, left_paths, right_paths, steering_angles, shift_steer)

    test_images = []
    for i in range(len(test_labels)):
        test_images.append(load_image(test_paths[i]))
    test_images = np.array(test_images)
            
    nb_train = len(train_labels)
    nb_val = len(validation_labels)

    nb_samples_per_epoch = 2 * nb_train * multiplier - 10 * batch_size
    nb_val_samples = 2 * nb_val * multiplier - 10 * batch_size
      
    # Then I create my generators
    train_generator = generate_batch(train_paths, train_labels, multiplier, shift_per_pix, batch_size)
    validation_generator = generate_batch(validation_paths, validation_labels, multiplier, shift_per_pix, batch_size)
                    
    # Then I define my model
    model = get_model()
    
    # Now I train your model here
    model.fit_generator(train_generator, samples_per_epoch=nb_samples_per_epoch, \
        nb_epoch=10, validation_data=validation_generator, nb_val_samples=nb_val_samples)
    
    # I save my model and weights           
    model.save_weights('model.h5')
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

    loss = model.evaluate(test_images, test_labels, batch_size = 128)
    print('Test Loss:' + str(loss))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

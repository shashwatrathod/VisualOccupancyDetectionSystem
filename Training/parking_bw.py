#Building the CNN

#importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image

#Initializing CNN
classifier = Sequential()
#Black and white version
#Step-1 Convolution
conv_layer_1 = Convolution2D(32, 3, 3, input_shape=(32, 32, 1), activation="relu")
classifier.add(conv_layer_1)

#Step-2 pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding second conv layer
conv_layer_2 = Convolution2D(32, 3, 3, input_shape=(32, 32, 1), activation="relu")
classifier.add(conv_layer_2)
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3 Flattening
classifier.add(Flatten())

#Step-4 Full Connection
classifier.add(Dense(output_dim = 128, activation="relu"))
classifier.add(Dense(output_dim = 1, activation="sigmoid"))

#Compiling the CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

#Fitting the CNN to dataset
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                        target_size=(32, 32),
                                                        batch_size=32,
                                                        color_mode = "grayscale",
                                                        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(32, 32),
        color_mode = "grayscale",
        batch_size=32,
        class_mode='binary')

from PIL import Image
classifier.fit_generator(
        training_set,
        steps_per_epoch=94720,
        epochs=15,
        validation_data=test_set,
        validation_steps=31825)

classifier.save("models/parking_bw_32_2.h5")
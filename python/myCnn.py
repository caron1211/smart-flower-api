from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import sys
import os
import urllib.request


import cnnModel
import prepareForCnn
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

path_test =  os.path.dirname(__file__) +"\\test" 
batch_size = 15


def loadModel():
    path_load = os.path.dirname(__file__) +"\\model.h5"
    model = cnnModel.load_trained_model(path_load, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    return model


def testModel(model):
    #  Prepare Testing Data
    test_df = prepareForCnn.prepareTest(path_test)
    nb_samples = test_df.shape[0]


    # Create Testing Generator
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        path_test,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    predict = model.predict(test_generator, steps=np.ceil(nb_samples / batch_size))
    a = np.argmax(predict, axis=-1)

    return a
    

def getTrainGenerator(train_df):

    #  Traning Generator
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        path_train,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    return train_generator

# debug
# print("1: " + sys.argv[1])
# print("2: " + path_test + "\\rose_02.jpg")
# url = "https://upload.wikimedia.org/wikipedia/commons/4/40/Sunflower_sky_backdrop.jpg"
# urllib.request.urlretrieve(url,"rose_02.jpg")


urllib.request.urlretrieve(sys.argv[1], path_test +"\\rose_02.jpg")

load_model = loadModel()

b = testModel(load_model)

print("prediction:{}".format(b[0]))
sys.stdout.flush()


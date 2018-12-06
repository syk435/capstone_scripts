from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
from keras.applications import VGG19
import numpy as np
import cv2
import tensorflow as tf

def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]


def to_multi_gpu(model, n_gpus=2):
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name="test_name")

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])

class VGG19Bottleneck:
    def __init__(self,width,height,train_data_dir,validation_data_dir):
        self.img_width = width
        self.img_height = height
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.nb_train_samples = 1024
        self.nb_validation_samples = 128
        self.epochs = 70
        self.batch_size = 16

    def train(self,opt,outputname):
        #if not using imagenet weights, switch off mean setting and do rescale=1/255
        datagen = ImageDataGenerator(rescale=1./255)
        model = VGG19(include_top=False, weights=None)
        #datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
        #datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)
        #model = VGG19(include_top=False, weights='imagenet')
        
        generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)
        bottleneck_features_train = model.predict_generator(
            generator, self.nb_train_samples // self.batch_size)
        #np.save(open('./models/bottleneck_features_train.npy', 'w'),bottleneck_features_train)
        
        generator = datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)
        bottleneck_features_validation = model.predict_generator(
            generator, self.nb_validation_samples // self.batch_size)
        #np.save(open('./models/bottleneck_features_validation.npy', 'w'),bottleneck_features_validation)

        #train_data = np.load(open('./models/bottleneck_features_train.npy'))
        train_data = bottleneck_features_train
        train_labels = np.array([0] * (284) + [1] * (284) + [2] * (259) + [3] * (197))

        #validation_data = np.load(open('./models/bottleneck_features_validation.npy'))
        validation_data = bottleneck_features_validation
        validation_labels = np.array([0] * (33) + [1] * (33) + [2] * (33) + [3] * (29))

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))
        model = to_multi_gpu(model,n_gpus=4)

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_data, train_labels,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_data=(validation_data, validation_labels))
        model.save_weights(outputname)

    def load(self,weights_path,opt):
        #if not using imagenet weights, switch off mean setting and do rescale=1/255
        datagen = ImageDataGenerator(rescale=1./255)
        model = VGG19(include_top=False, weights=None)
        #datagen = ImageDataGenerator(rescale=1., featurewise_center=True) #(rescale=1./255)
        #datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)
        #model = VGG19(include_top=False, weights='imagenet')
        generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)
        train_data = model.predict_generator(
            generator, self.nb_train_samples // self.batch_size)
        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))
        model = to_multi_gpu(model,n_gpus=4)
        model.load_weights(weights_path)
        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
        self.model = model

    def predict(self,images):
        vgg_model = VGG19(weights=None, include_top=False)
        #vgg_model = VGG19(weights='imagenet', include_top=False )
        bottom_features = vgg_model.predict(images)
        return self.model.predict(bottom_features,batch_size=32, verbose=1)

    def evaluate(self):
        # print classes
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = test_datagen.flow_from_directory(
                self.validation_data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=16,
                class_mode='categorical')
        print('\nClasses: ', validation_generator.class_indices)

        # evaluate with generated validation data
        print('Evaluating Model with generator:\n','Model Metrics: ', self.model.metrics_names)
        evals = self.model.evaluate_generator(validation_generator, 40)
        print('Model Evaluation: ',evals)

    def clear_session(self):
        # del session to supress tensorflow error
        K.clear_session()

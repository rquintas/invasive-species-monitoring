import argparse
import datetime
import json

import math

import os

import numpy as np
import pandas

import keras
from keras import optimizers, applications, Input, regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ZeroPadding2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator


def model_inception_v3_finetune_more_layers(img_width, img_height):

    base_model = applications.InceptionV3(include_top=False, weights='imagenet',
                                          input_shape=(3, img_width, img_height))

    for layer in base_model.layers:
        layer_name = str(layer)
        if ('Conv2D' not in layer_name) and ('BatchNormalization' not in layer_name):
            layer.trainable = False

    top_model = Sequential(name='top_layer')

    top_model.add(Flatten(name='flat', input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', name='dense1', kernel_regularizer=regularizers.l2(5e-4)))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(256, activation='relu', name='dense2', kernel_regularizer=regularizers.l2(5e-4)))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid', name='sigmoid'))

    optimizer = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model = Model(input=base_model.input, output=top_model(base_model.output), name='inceptionv3')

    model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    return model

if __name__ == '__main__':
    startts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = argparse.ArgumentParser(description='Learn to identify invasive species from photos.')

    parser.add_argument('--id', dest='id', type=str,
                        help='run identifier',
                        default=startts)

    parser.add_argument('--splitnum', dest='splitnum', type=int,
                    help='split to use',
                    default=1)

    # Setup arguments              
    args = parser.parse_args()

    id = args.id
    splitnum = args.splitnum

    # Setup folders
    os.chdir('data')

    train_data_dir = 'data_5_splits_augmented_256_256/split_{}/train'.format(splitnum)
    validation_data_dir = 'data_5_splits_augmented_256_256/split_{}/validation'.format(splitnum)
    test_data_dir = 'data_test_augmented_256_256'

    ###################
    # Create model.
    model = model_inception_v3_finetune_more_layers(256, 256)

    # Setup callbacks, use early stopping and save best models to disk.
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint('{0}.fold_{1}.model'.format(id, splitnum), monitor='val_acc', save_best_only=True)
    callbacks = [earlystop, checkpoint]

    ###################
    # Train

    n_epochs = 5000

    train_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

    print('Train samples: {} steps: {}, Validation samples: {} steps: {}'
    .format(train_generator.samples, 
    int(math.ceil(float(train_generator.samples) / 32)),
    validation_generator.samples, 
    int(math.ceil(float(validation_generator.samples) / 32))
    ))

    history = model.fit_generator(
        train_generator,
        int(math.ceil(float(train_generator.samples) / 32)),
        validation_data=validation_generator,
        validation_steps=int(math.ceil(float(validation_generator.samples) / 32)),
        callbacks=callbacks,
        epochs=n_epochs,
        verbose=2)

    # Reload best weights found.
    model.load_weights('{0}.fold_{1}.model'.format(id, i))

    ###################
    # Test
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(256, 256),
        batch_size=32,
        shuffle=False,
        class_mode='binary')

    preds = model.predict_generator(test_generator, int(math.ceil(float(test_generator.samples) / 32)))

    ###################
    # Save run to json file.
    with open(args.id + '.json', 'w') as outfile:

        info = {'sanity': False,
                'history': [history.history],
                'balanced': True,
                'sample': False,
                'model': 'inception v3 finetune',
                'id': args.id,
                'imgsize': '256x256',
                'train_loss': [],
                'validation_loss': [],
                'nfolds': 5,
                'prediction_augmented': True}

        json.dump(info, outfile, indent=4)

    ###################
    # Create submission
    filenames = test_generator.filenames
    
    # Remove filename suffixes of augmented files.
    ids = np.array([int(f.replace('0.5/','')
                        .replace('.0','')
                        .replace('_0','')
                        .replace('_1','')
                        .replace('_2','')
                        .replace('_3','')
                        .replace('_4','')
                        .replace('.png','')) for f in filenames])
    invasive = preds[:,0]
    
    subm = np.stack([ids,invasive], axis=1)

    submission_file_name = args.id+'-submission.csv'

    # Average the augmentation predictions.
    df = pandas.DataFrame(data=subm, columns=['name','invasive'])
    df.groupby('name').mean().to_csv(submission_file_name, float_format="%.04f")
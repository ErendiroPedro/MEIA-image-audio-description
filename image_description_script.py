# -*- coding: utf-8 -*-

#### Import dependencies ####
from my_utils import *

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model

import time
import datetime
import pickle

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, TimeDistributed, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split


# 

### GLOBAL VARIABLES ###
CWD = os.getcwd()
CAPTION_FOLDER = 'captions'
CAPTIONS_FILE_NAME = "labels.csv"
IMAGE_FOLDER = 'images'
PRE_LOADING_FOLDER = 'pre_loaded_data'
TIMESTAMP ='{date:%Y%m%d%H%M}'.format(date=datetime.datetime.now())
EXECUTION_KEY = TIMESTAMP + randomString() 

### PRE LOADED DATA ###
PP_CAPTION_FILE_NAME = "202304050611ZMBVM_pp_captions.npy"
PP_TOKENIZER_NAME = "202304050611ZMBVM_tokenizer.p"
PP_IMAGE_FILE_NAME = "202304050559CTGID_pp_images.npy"
PP_FEATURE_FILE_NAME = "202304050619ABSSJ_features.npy"


#### FUNCTIONS ####
def load_and_pre_process_captions(load_data_from_file):
    start_time = time.time()
    
    if(load_data_from_file):
        pp_captions = np.load(os.path.join(CWD,PRE_LOADING_FOLDER, PP_CAPTION_FILE_NAME),allow_pickle=True)
        print("*** Loaded pre-processed captions from", PP_CAPTION_FILE_NAME,"***" )
        
        with open(os.path.join(CWD,PRE_LOADING_FOLDER, PP_TOKENIZER_NAME), 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("*** Loaded pre-processed tokenizer from", PP_TOKENIZER_NAME,"***" )
        return pp_captions, tokenizer
    

    captions_file_path = os.path.join(CWD, CAPTION_FOLDER)
    captions_file = os.path.join(captions_file_path, CAPTIONS_FILE_NAME)
    captions_df = pd.read_csv(captions_file,delimiter='|', lineterminator='\n')
    captions_list = captions_df['comment'].to_list()


    ## Processing
    pp_captions, tokenizer = process_caption(captions_list)
    
    
    ## Logging and saving
    end_time = time.time()
    elapsed_time = end_time - start_time
    

    print("*** Saved preprocessed caption data into file:",EXECUTION_KEY + "_pp_captions" ,"***")
    file_name = os.path.join(CWD,PRE_LOADING_FOLDER, EXECUTION_KEY + "_pp_captions")
    np.save(file_name, pp_captions)
    print("*** Saved preprocessed tokenizer data into file:",EXECUTION_KEY + "_tokenizer" ,"***")
    file_name = os.path.join(CWD,PRE_LOADING_FOLDER, EXECUTION_KEY + "_tokenizer.p")
    with open(file_name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("*** Captions loaded and processed (", elapsed_time/60,") ***")
    return pp_captions, tokenizer

def load_and_pre_process_images(load_data_from_file):  
    start_time = time.time() 
    
    if(load_data_from_file):
        # Load pre processed captions from file
        pp_images = np.load(os.path.join(CWD,PRE_LOADING_FOLDER, PP_IMAGE_FILE_NAME))
        print("*** Loaded pre-processed images from", PP_IMAGE_FILE_NAME,"***" )
        return pp_images
    
    images_folder_path = os.path.join(CWD, IMAGE_FOLDER)
    images_data = {} 
    for file_name in os.listdir(images_folder_path):
        # check if the file is an image by checking the file extension
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            image_path = os.path.join(images_folder_path, file_name)
            pp_image = process_image(image_path)
            images_data[file_name] = pp_image
            
    captions_file_path = os.path.join(CWD, CAPTION_FOLDER, CAPTIONS_FILE_NAME)
    captions_df = pd.read_csv(captions_file_path,delimiter='|', lineterminator='\n')

    pp_images = np.array([images_data[image_id] for image_id in captions_df.image_name])


    end_time = time.time()
    elapsed_time = end_time - start_time
    

    print("*** Saved pre-processed caption data into file:",EXECUTION_KEY + "_pp_images" ,"***")
    file_name = os.path.join(CWD,PRE_LOADING_FOLDER, EXECUTION_KEY + "_pp_images")
    np.save(file_name, pp_images)
    print("*** Images loaded and processed (", elapsed_time/60,") ***")
    return pp_images

def extract_features_using_CNN(pp_images, load_data_from_file):
    start_time = time.time()

    if(load_data_from_file):
        # Load pre processed captions from file
        features = np.load(os.path.join(CWD,PRE_LOADING_FOLDER, PP_FEATURE_FILE_NAME))
        print("*** Loaded preprocessed features from", PP_FEATURE_FILE_NAME,"***" )
        return features
    
    # Load the VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SZ,IMG_SZ,3))
    
    print("--- Retrieving features from images ---")
    features = base_model.predict(pp_images)
    

    """
    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False
        
    # create a new model that uses the base model
    model = Sequential()
    model.add(base_model)
    
    # Add more layers to increase the complexity of the model
    model.add(Conv2D(IMG_SZ, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(IMG_SZ, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(MAX_CAPTION_SIZE, activation='softmax'))
    
    # compile and train the new model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(pp_images, pp_captions, epochs=epochs, verbose=1)
    


    file_name = os.path.join(CWD,PRE_LOADING_FOLDER, TIMESTAMP + EXECUTION_KEY + "_cnn_model.h5")
    model.save(filename)
    print("*** Saved trained model to file",filename,"***")
    plt.plot(model.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    """
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("*** CNN model trained (", elapsed_time/60,"minutes passed) ***")
    
    print("*** Saved image features to file", EXECUTION_KEY + "_features.npy","***")
    file_name = os.path.join(CWD,PRE_LOADING_FOLDERID, ID + "_features.npy")
    np.save(file_name,features)
    
    return features

def build_captioning_model(features, num_words, max_len, embedding_dim, dropout_rate):
    """
    Builds a CNN-RNN image captioning model given the image features and other required parameters.

    Args:
    - features (numpy.ndarray): Image features extracted by a pre-trained CNN.
    - num_words (int): Number of unique words in the vocabulary.
    - max_len (int): Maximum length of a caption (in words).
    - embedding_dim (int): Dimensionality of the word embeddings.
    - lstm_units (int): Number of LSTM units in the decoder.
    - dropout_rate (float): Dropout rate used in the decoder.

    Returns:
    - model (tensorflow.keras.Model): A compiled Keras model for image captioning.
    """
    
    lstm_units=features.shape[3]
    
    # Decoder architecture
    inputs1 = Input(shape=(features.shape[1],))
    fe1 = Dropout(dropout_rate)(inputs1)
    fe2 = Dense(lstm_units, activation='relu')(fe1)

    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(num_words, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(dropout_rate)(se1)
    se3 = LSTM(lstm_units)(se2)

    decoder1 = Concatenate()([fe2, se3])
    decoder2 = Dense(lstm_units, activation='relu')(decoder1)
    outputs = Dense(num_words, activation='softmax')(decoder2)

    # Encoder architecture
    inputs3 = Input(shape=features.shape[1:])
    fe3 = Dropout(dropout_rate)(inputs3)
    
    fe4 = Dense(lstm_units, activation='relu')(fe3)

    # Define the full model by combining the encoder and decoder
    decoder3 = fe4
    decoder4 = decoder2
    inputs = [inputs1, inputs2, inputs3]
    outputs = decoder4
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = RMSprop(lr=0.001, clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    # summarize model
    print(model.summary())
    plot_model(model, to_file=EXECUTION_KEY + '_model.png', show_shapes=True)
    
    return model

def main():
    
    print("--- Hi, the program has started ---")
    
    #### Pre-processing ####     
    pp_captions, tokenizer = load_and_pre_process_captions(load_data_from_file=True)
    ## DEBUG
    print("pp_captions type:", type(pp_captions))
    print("pp_captions shape:", pp_captions.shape)
    # print ("pp_captions:", pp_captions)

    pp_images = load_and_pre_process_images(load_data_from_file=True)
    ## DEBUG
    print("pp_images type:", type(pp_images))
    print("pp_images shape:", pp_images.shape)
    # print("pp_images: ", pp_images)
    
    # Split the data into training and test sets (30% of the data used for testing)
    images_for_training, images_for_testing, captions_for_training, caption_for_testing = train_test_split(pp_images, pp_captions, test_size=0.6)
    ## DEBUG
    print("images_for_training  len:", len(images_for_training))
    print("captions_for_training len:", len(captions_for_training))

    #### Machine learning ####  
    
    ## Training ##
    features_for_training = extract_features_using_CNN(images_for_training, load_data_from_file=True) 
    ## DEBUG
    print("features_for_training  len:", len(features_for_training))
    print("features_for_training shape:", features_for_training.shape)
    
    num_words = len(tokenizer.word_index)
    max_len = len(max(pp_captions, key=len))
    embedding_dim = 100
    dropout_rate = 0.5
    model = build_captioning_model(features_for_training, num_words, max_len, embedding_dim, dropout_rate)
    

    epochs = 10
    steps = len(captions_for_training)
    
    for i in range(epochs):
        generator = data_generator(captions_for_training, features_for_training, tokenizer, max_len)
        model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
        model.save("models/model_" + str(i) + ".h5")

    
    ## Validation ##  

#### START PROGRAM #### 

main()







"""


# Preprocess the caption, splitting the string and adding <start> and <end> tokens
def get_preprocessed_caption(caption): 
    print(caption)
    caption = re.sub(r'\s+', ' ', caption)
    caption = caption.strip()
    caption = "<start> " + caption + " <end>"
    return caption


for index, image_i in df.iterrows(): # original feature: image_name, comment_number, comment
    print(get_preprocessed_caption(image_i['comment']))



"""
    
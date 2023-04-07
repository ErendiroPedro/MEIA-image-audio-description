# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:23:28 2023

@author: espedro
"""

import numpy as np
import tensorflow as tf

from PIL import Image

from random import choice
import string


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import constant, concat

from keras.utils import to_categorical

### GLOBAL VARIABLES ###
IMG_SZ = 32
IMAGE_SIZE = (IMG_SZ, IMG_SZ)
INPUT_SHAPE = (IMG_SZ,IMG_SZ,3)



def randomString(stringLength=5):
    letters = string.ascii_uppercase
    return ''.join(choice(letters) for i in range(stringLength))

def normalize_list(arr_list):
    normalized_list = []
    for lst in arr_list:
        epsilon = 1e-8
        normalized_lst = (lst - np.min(lst)) / (np.max(lst) - np.min(lst) + epsilon)
        normalized_list.append(normalized_lst)

    return np.array(normalized_list, dtype=object)

def process_image (image_path):
    
    # Load an image from file
    image = Image.open(image_path)
    
    ## 1 - Resize image
    image = image.resize(IMAGE_SIZE)
    
    ## 2 - Convert to array
    image_array = np.array(image)
    
    ## 3 - Normalize pixel values
    normalized_image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
    ## ADD MORE

    return normalized_image_array

def process_caption (captions_list):
    
    ## DEBUG
    #print("size of raw captions:", len(captions_list))
    #print("type raw captions:", type(captions_list))
    #print("raw captions:", captions_list)
    
    
    ### 1 - Clean captions
    clean_captions =[]
    for caption in captions_list:
        if str(caption) == 'nan':
            caption = "<UKN>"
            clean_captions.append(caption)
            continue
            
        # delete digits, special chars, etc., 
        caption = caption.replace('[^A-Za-z]', '')
        # convert to lowercase
        caption = caption.lower()
        # delete additional spaces
        caption = caption.replace('\s+', ' ')
        # add start and end tags to the caption
        caption = '<SOS> ' + " ".join([word for word in  caption.split() if len(word)>1]) + ' <EOS>'
        #print(caption)
        clean_captions.append(caption)
     
    ## DEBUG
    #print("size of clean captions:", len(clean_captions))
    #print("type clean captions:", type(clean_captions))
    #print("clean captions", clean_captions)
    
    
    ### 2 - Tokenize captions and encode sequence
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(clean_captions)
    
    captions_as_sequences = tokenizer.texts_to_sequences(clean_captions)

    len_longest_caption = len(max(captions_as_sequences, key=len))
    padded_sequences = pad_sequences(captions_as_sequences, maxlen=len_longest_caption, padding='post')
    
    
    ## 3 - Normalize padded sequences
    normalized_captions_as_sequences = np.array(normalize_list(padded_sequences))
 
    
    ## DEBUG
    #print("size of pp captions:", len(normalized_captions_as_sequences))
    #print("type pp captions:", type(normalized_captions_as_sequences))
    #print("pp captions", normalized_captions_as_sequences)
    
    return normalized_captions_as_sequences, tokenizer

#data generator, used by model.fit_generator()
def data_generator(captions_list, features, tokenizer, max_length):
    while 1:
        for feature,caption in zip(features,captions_list):
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, captions_list, feature)
            yield [[input_image, input_sequence], output_word]
def create_sequences(tokenizer, max_length, caption_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for caption in caption_list:
        # split one sequence into multiple X,y pairs
        for i in range(1, len(caption)):
            # split into input and output pair
            in_seq, out_seq = caption[:i], caption[i]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index))[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

"""
## DECODING ##
def train_RNN(pp_images, pp_captions, epochs, word_index):
    input_seq = Input(shape=(None,))
    embedded_seq = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_seq)
    x = LSTM(units=256, return_sequences=True)(embedded_seq)
    x = LSTM(units=256)(x)
    x = Dropout(0.5)(x)
    output = Dense(units=vocab_size, activation='softmax')(x)
    rnn_model = Model(inputs=input_seq, outputs=output)


# How to improve detection
def write_number_of_people (image_data, labels_df):
    
    number_of_people_dict = {}
    path = os.path.join(os.getcwd(), 'test_images')
    labels_df = labels_df.reset_index()  # make sure indexes pair with number of rows
    new_feature = []
    
    for file_name, pp_image in image_data:
        # Load the pre-trained SSD model
        model = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
        
        # Create a blob from the image
        image_path = os.path.join(path, file_name)
        print(image_path)
        cv2_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(cv2_image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        
        # Set the input of the model to the blob
        model.setInput(blob)
        
        # Forward pass through the model to get the output
        output = model.forward()
        
        num_people = 0
        # Loop through the output
        for detection in output[0,0,:,:]:
            confidence = detection[2]
            
            # If the confidence is greater than 0.5
            if confidence > 0.5:
                class_id = detection[1]
                
                # If the class is person
                if class_id == 15:
                    # Get the coordinates of the bounding box
                    left = int(detection[3] * cv2_image.shape[1])
                    top = int(detection[4] * cv2_image.shape[0])
                    right = int(detection[5] * cv2_image.shape[1])
                    bottom = int(detection[6] * cv2_image.shape[0])
                    
                    # Draw the bounding box on the image
                    cv2.rectangle(cv2_image, (left, top), (right, bottom), (0, 255, 0), 2)
                    plt.imshow(cv2_image, interpolation='nearest')
                    plt.show()
                    # Increment the counter for the number of people
                    num_people += 1
                    
        number_of_people_dict[file_name] = num_people
    
    for index, row in labels_df.iterrows():
        file_name = row['image_name']
        if file_name in number_of_people_dict:
            number_of_people = number_of_people_dict[file_name]
            new_feature.append(number_of_people) # Add number of people in each image to data frame
    
    # print(new_feature)
    labels_df ["number_of_people"] = new_feature
    print("New feature 'number_of_people' added...")
    return labels_df

 #### Feature exploration ####
 # Add feature "number_of_people"
 # main_df = write_number_of_people(image_data, main_df)
 #num_unique_classes=len(np.unique(captions_for_training))
 #print(num_unique_classes)
"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import time
import pickle
import sklearn
import cvlib as cv
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt



from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.applications import ResNet50, InceptionResNetV2, MobileNetV2, VGG16
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50,InceptionResNetV2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from skmultilearn.model_selection import IterativeStratification
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import classification_report 
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from keras.utils import to_categorical


# In[27]:


# Function to load the pre-trained soft voting model using pickle
def load_soft_voting_model(model_file_path):
    with open(model_file_path, 'rb') as model_file:
        soft_voting_model = pickle.load(model_file)
    return soft_voting_model['model']

InceptionResNetV2_model_file_path = "C:/Users/Project/Desktop/DeepFake_Video_Detection/DeepFake_Video_Detection/DeepFake_Video_Detection/deep_fake-html/MobileNetV2_model_history.pkl"
# Load the soft voting model
loaded_InceptionResNetV2_model = load_soft_voting_model(InceptionResNetV2_model_file_path)


# In[28]:


def process_single_video(video_path, frameTime):
    # Initialize lists to store frame data, video IDs, and frame IDs
    ListFrames = []
    video_ids = []
    frame_ids = []

    total_videos = 0

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the video's frame rate
    frame_interval = int(frame_rate * frameTime)  # Calculate the frame capture interval

    frame_counter = 0
    video_frame_ids = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break if there are no more frames

        frame_counter += 1

        # Capture frames at the specified interval
        if frame_counter % frame_interval == 0:
            frame = cv2.resize(frame, (128, 128))  # Resize frames to 128x128
            ListFrames.append(frame)
            video_ids.append(total_videos)
            frame_ids.append(video_frame_ids)
            video_frame_ids += 1  # Increment frame ID for the video

    cap.release()  # Release video capture object after processing

    # Create a DataFrame from collected data
    data = {
        'VideoID': video_ids,
        'FrameID': frame_ids,
        'Frames': ListFrames,
    }
    df = pd.DataFrame(data)

    # Print summary information
    # print(f"Capture one frame every {frameTime} seconds")
    # print(f"Total number of frames: {len(df)}")

    return df  # Return the DataFrame containing video frame information

# Example usage
# video_path = "C:/Users/Project/Desktop/fake.mp4"
# frame_time_seconds = 2
# processed_df = process_single_video(video_path, frame_time_seconds)


# In[29]:


# processed_df


# In[30]:


def normalize_frames(df):
    np_ListFrames = np.array(df['Frames'])

    # Normalize the pixel values to be in the range [0, 1]
    np_ListFrames = np_ListFrames / 255.0

    # Add a new column to the existing DataFrame
    df['Normalized Frames'] = list(np_ListFrames)

    return df

# # Example usage
# normalized_df = normalize_frames(processed_df)


# In[8]:


#normalized_df=normalized_df[['VideoID','FrameID','Normalized Frames']]
#normalized_df


# In[31]:


def classify_frames(df, model):
    np_normalized_frames = np.array(df['Normalized Frames'])
    predictions = []

    # Iterate through each normalized frame and make predictions
    for normalized_frame in np_normalized_frames:
        X_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
        prediction = model.predict(X_frame)[0]  # Assuming binary output
        predictions.append(prediction)

    # Add the predictions to the DataFrame
    df['Predictions'] = predictions

    return df

# # Example usage
# # Assuming 'your_pretrained_model' is the pre-trained model for real/fake classification
# classified_df = classify_frames(normalized_df, loaded_InceptionResNetV2_model)


# In[32]:


# classified_df


# In[33]:


# # Apply argmax to 'Predictions' column and append the result as a new column
# classified_df['PredictedLabels'] = np.argmax(classified_df['Predictions'].tolist(), axis=1)
# # Display the modified DataFrame
# print(classified_df)


# In[34]:


# # Calculate the total number of frames for each video
# video_frame_counts = classified_df['VideoID'].value_counts()

# # Iterate through unique video IDs
# for vid in classified_df['VideoID'].unique():
#     video_subset = classified_df[classified_df['VideoID'] == vid]

#     # Count the occurrences of predicted labels
#     count_real = np.sum(video_subset['PredictedLabels'] == 1)  # Assuming 1 corresponds to 'Real' in the model

#     # Calculate the percentage of 'Real' frames
#     percentage_real = (count_real / video_frame_counts[vid]) * 100

#     # Determine the model decision ('Real' or 'Fake') based on the percentage of 'Real' frames
#     if percentage_real >= 70:
#         result = 'Real'
#         # Display the final results for 'Real' videos
#         print(f"The video is {result} by {percentage_real}%")
#     else:
#         result = 'Fake'
#         # Display the final results for 'Fake' videos
#         print(f"The video is {result}  by {100 - percentage_real}%")


# In[38]:


def process_classify_video(video_path_new , frame_time_seconds=2):
    processed_df = process_single_video(video_path_new, frame_time_seconds)
    normalized_df = normalize_frames(processed_df)
    classified_df = classify_frames(normalized_df, loaded_InceptionResNetV2_model)
    classified_df['PredictedLabels'] = classified_df['Predictions'].apply(lambda x: np.argmax(x))
    
    video_frame_counts = classified_df['VideoID'].value_counts()

    # Iterate through unique video IDs
    for vid in classified_df['VideoID'].unique():
        video_subset = classified_df[classified_df['VideoID'] == vid]

        # Count the occurrences of predicted labels
        count_real = np.sum(video_subset['PredictedLabels'] == 1)  # Assuming 1 corresponds to 'Real' in the model

        # Calculate the percentage of 'Real' frames
        percentage_real = (count_real / video_frame_counts[vid]) * 100

        # Determine the model decision ('Real' or 'Fake') based on the percentage of 'Real' frames
        if percentage_real >= 70:
            result = 'Real'
            # Display the final results for 'Real' videos
            return (f"The video is {result} by {percentage_real}%")
        else:
            result = 'Fake'
            # Display the final results for 'Fake' videos
            return (f"The video is {result}  by {100 - percentage_real}%")


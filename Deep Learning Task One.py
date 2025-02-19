#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras


# In[2]:


train_path = r'C:\Users\PC\Downloads\Dataset\train'
test_path = r'C:\Users\PC\Downloads\Dataset\test'
val_path = r'C:\Users\PC\Downloads\Dataset\val'


# In[3]:


from keras.preprocessing.image import ImageDataGenerator


# In[4]:


train_batches= ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet.preprocess_input).flow_from_directory( directory=train_path, target_size=(224,224), batch_size= 10)
test_batches= ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet.preprocess_input).flow_from_directory( directory=test_path, target_size=(224,224), batch_size= 10, shuffle = False )
val_batches= ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet.preprocess_input).flow_from_directory( directory=val_path, target_size=(224,224), batch_size= 10)


# In[5]:


imgs,labels= next(train_batches)


# In[6]:


vgg16_model = tf.keras.applications.vgg16.VGG16()


# In[7]:


model = keras.Sequential()


# In[8]:


for layer in vgg16_model.layers:
    layer.trainable = False  
    model.add(layer)


# In[12]:


model.add(keras.layers.Flatten())  
model.add(keras.layers.Dense(256, activation='relu'))  
model.add(keras.layers.Dense(2, activation='softmax'))  


# In[13]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[14]:


model.fit(train_batches, validation_data=val_batches, steps_per_epoch=4, validation_steps=4, epochs=5, verbose=2)


# In[16]:


loss, acc = model.evaluate(test_batches)
print(f"Test Accuracy: {acc:.2f}")


# In[19]:


from tensorflow.keras.preprocessing import image
img_path = r'C:\Users\PC\Downloads\Dataset\test\cat\832.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)


# In[20]:


prediction = model.predict(img_array)
print("Prediction:", prediction)


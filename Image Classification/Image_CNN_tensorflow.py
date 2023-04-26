#!/usr/bin/env python
# coding: utf-8

# # import Libraries

# In[1]:


import tensorflow as tf
import os
import numpy as np
import cv2
import imghdr
import matplotlib
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# # Define the Dataset class

# In[230]:


DataDir = 'image data'
ImageEx = ['jpeg', 'jpg', 'bmp', 'png']


# In[231]:


print(os.listdir(DataDir))
print(os.listdir(os.path.join(DataDir, 'Beyonce',)))


# In[232]:


for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir,image_class)):
        print(image)

total_images = 0  # initialize counter variable
for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir, image_class)):
        total_images += 1  # increment counter for each file
print("Total number of images:", total_images)


# In[233]:


total_images = 0  # initialize counter variable
removed_images = 0  # initialize counter variable for removed images
for image_class in os.listdir(DataDir):
    for image in os.listdir(os.path.join(DataDir, image_class)):
        ImagePath = os.path.join(DataDir, image_class,image)
        try:
            DImage = cv2.imread(ImagePath)
            ImaS = imghdr.what(ImagePath)
            if ImaS not in ImageEx:
                print('Image not in extension list {}'.format(ImagePath))
                os.remove(ImagePath)
                removed_images += 1  # increment counter for each removed file
        except Exception as e:
            print('Issue with image {}'.format(ImagePath))
            # os.remove(ImagePath)  # Uncomment this if you want to remove the problematic images
        total_images += 1  # increment counter for each file


# In[234]:


print("Total number of images found:", total_images)


# In[235]:


print("Total number of images removed:", removed_images)


# # preprocessing the images

# In[236]:


ImageData = tf.keras.utils.image_dataset_from_directory('image data', )
print(ImageData)


# In[237]:


# data image iterator is ImageItr
ImageItr = ImageData.as_numpy_iterator()
try:
    Imagebatch = ImageItr.next()
    print(Imagebatch[1].shape)
except StopIteration:
    pass


# In[238]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))

for idx, DImage in enumerate(Imagebatch[0][:4]):
        ax[idx].imshow(DImage.astype(int))
        ax[idx].title.set_text(Imagebatch[1][idx])

plt.show()


# #preprocessing the data

# In[239]:


ImageData = tf.keras.utils.image_dataset_from_directory('Image data')


# In[240]:


ImageData = ImageData.as_numpy_iterator().next()


# 

# In[241]:


x_train, y_train = ImageData
ImageData = tf.data.Dataset.from_tensor_slices((x_train / 255, y_train))


# In[242]:


ImageData.as_numpy_iterator().next()


# # Check the range of the data after scaling

# In[243]:


print(ImageData.as_numpy_iterator().next()[0].max())
print(ImageData.as_numpy_iterator().next()[0].min())


# # Display a few images from the dataset

# In[244]:


scaled_iterator = ImageData.as_numpy_iterator()
Thebatch = scaled_iterator.next()


# # set the sizes for the train, validation, and test sets

# In[245]:


TrainSize = int(len(ImageData)*1)-10
ValSize = int(len(ImageData)*.0)+7
TestSize = int(len(ImageData)*.0)+3


# # load the image dataset

# In[246]:


Idata = tf.keras.preprocessing.image_dataset_from_directory(
    'Image data',
    batch_size=8,
    image_size=(256, 256),
    validation_split=0.3,
    subset='training',
    seed=123
)


# In[247]:


print('Training Size=', TrainSize)


# In[248]:


print('Validation Size=', ValSize)


# In[249]:


print('Test Size=', TestSize)


# In[250]:


print('Total Size=', TrainSize + ValSize + TestSize)


# In[251]:


# the skip and take tensorflow function is used so the data from the training won't appear in the validation or test set
Train = Idata.take(TrainSize)
Val = Idata.skip(TrainSize).take(ValSize)
Test = Idata.skip(TrainSize+ValSize).take(TestSize)


# In[252]:


len(Train)


# In[253]:


len(Val)


# In[254]:


len(Test)


# # building the model

# In[255]:


IModel = Sequential()
IModel.add(Conv2D(8, (3, 3), 2, activation='relu', input_shape=(256, 256, 3)))
IModel.add(MaxPooling2D())
IModel.add(Conv2D(16, (3, 3), 2, activation='relu'))
IModel.add(MaxPooling2D())
IModel.add(Conv2D(8, (3, 3), 2, activation='relu'))
IModel.add(MaxPooling2D())
IModel.add(Flatten())
IModel.add(Dense(256, activation='relu'))
IModel.add(Dense(4, activation='softmax'))


# In[256]:


# i used less filter for it to be faster at the expense of accuracy and the stride of 2 instead of 1 for speed over accuracy
# If you have 4 classes of images to classify, you should use Dense(4, activation='softmax') instead of Dense(1, activation='sigmoid') in your final layer.
# The softmax activation function is typically used for multi-class classification problems and produces a probability distribution over the different classes.
# The output of the softmax layer will be a vector of length 4, with each element representing the probability of the corresponding class.


# # compile model

# In[257]:


IModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# # print model summary

# In[258]:


print(IModel.summary)


# In[259]:


LogDir = 'ImageLogs'


# In[260]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LogDir)


# In[261]:


print(Val)


# In[262]:


hist = IModel.fit(Train, epochs=35, validation_data= Val, callbacks=[tensorboard_callback])


# In[263]:


hist.history


# # Graph of the history

# In[267]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='blue', label='loss')
plt.plot(hist.history['val_loss'], color='green', label='val_loss')
fig.suptitle('LOSS OF TRAIN AND VAL', fontsize=30)
plt.legend(loc="upper left")
plt.show()


# In[268]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='blue', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='green', label='val_accuracy')
fig.suptitle('ACCURACY OF TRAIN AND VAL', fontsize=30)
plt.legend(loc="upper left")
plt.show()


# # EVALUATION

# In[266]:


# evaluate the model on the test set
Test_loss, Test_accuracy = IModel.evaluate(Test)

print("Test Loss:", Test_loss)
print("Test Accuracy:", Test_accuracy)


# # SAVE THE MODEL

# In[273]:


from tensorflow.keras.models import load_model


# In[275]:


IModel.save(os.path.join('Models','ImageClassification.h5'))


# # SAVE THE CLASSES

# In[277]:


train_dir = 'Image data'
classes = sorted(os.listdir(train_dir))


# In[279]:


print(classes)


# In[278]:


class_dict = {}
for i, class_label in enumerate(classes):
    class_dict[class_label] = i


# In[280]:


import json
with open('class_dict.json', 'w') as f:
    json.dump(class_dict, f)


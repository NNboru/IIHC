#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install opencv-python==3.4.4.19


# In[ ]:


#Spectrograms
from glob import glob
import keras
from sklearn.model_selection import train_test_split
import fnmatch
import cv2
import numpy as np



#viterbi
from glob import glob
import keras
from sklearn.model_selection import train_test_split
import fnmatch
import cv2
import numpy as np


print('viterbi')
directory = "/scratch/sks.cse.iitbhu/edf/viterbi/eval"
gpath = directory+'/**/*.png'
imagePatches = glob(gpath, recursive=True)
for filename in imagePatches[0:20]:
    print(filename)

import fnmatch
patternZero = '*/0/*.png'
patternOne = '*/1/*.png'
classZero = fnmatch.filter(imagePatches, patternZero)
classOne = fnmatch.filter(imagePatches, patternOne)
print("IDC(-)\n\n",classZero[0:5],'\n')
print("IDC(+)\n\n",classOne[0:5])

def proc_images(lowerIndex,upperIndex):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    x = []
    y = []
    WIDTH = 256
    HEIGHT = 256
    for img in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y

X,Y = proc_images(0,300000)

from keras.utils.np_utils import to_categorical
import keras
from sklearn.model_selection import train_test_split

num_classes=2
X=np.array(X)
Y=np.array(Y)



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)

Y_traincat = keras.utils.to_categorical(Y_train, num_classes)
Y_valcat = keras.utils.to_categorical(Y_val, num_classes)


print("Training Features:", X_train.shape)
print("Training Labels:", Y_train.shape)
print("Validation Features:", X_val.shape)
print("Validation Labels:", Y_val.shape)
print(Y.shape)


saveX_train = directory+'/X_train.npy'
saveY_train = directory+'/Y_train.npy'

saveX_val = directory+'/X_val.npy'
saveY_val = directory+'/Y_val.npy'

saveY_train1 = directory+'/Y_train_1dim.npy'
saveY_val1 = directory+'/Y_val_1dim.npy'

np.save(saveX_train,X_train)
np.save(saveY_train,Y_traincat)

np.save(saveX_val,X_val)
np.save(saveY_val,Y_valcat)

np.save(saveY_train1,Y_train)
np.save(saveY_val1,Y_val)
        
directory = "/scratch/sks.cse.iitbhu/edf/viterbi/train"
gpath = directory+'/**/*.png'
imagePatches = glob(gpath, recursive=True)
for filename in imagePatches[0:20]:
    print(filename)

import fnmatch
patternZero = '*/0/*.png'
patternOne = '*/1/*.png'
classZero = fnmatch.filter(imagePatches, patternZero)
classOne = fnmatch.filter(imagePatches, patternOne)
print("IDC(-)\n\n",classZero[0:5],'\n')
print("IDC(+)\n\n",classOne[0:5])

def proc_images(lowerIndex,upperIndex):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    x = []
    y = []
    WIDTH = 256
    HEIGHT = 256
    for img in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y

X,Y = proc_images(0,300000)

from keras.utils.np_utils import to_categorical
import keras
from sklearn.model_selection import train_test_split

num_classes=2
X=np.array(X)
Y=np.array(Y)





Y1 = keras.utils.to_categorical(Y, num_classes)


print("Training Features:", X.shape)
print("Training Labels:", Y1.shape)


saveX = directory+'/X_test.npy'
saveY = directory+'/Y_test.npy'
saveY_test1 = directory+'/Y_test_1dim.npy'



np.save(saveX,X)
np.save(saveY,Y1)
np.save(saveY_test1,Y)


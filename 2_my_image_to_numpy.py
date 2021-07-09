from glob import glob
import cv2
from sklearn.model_selection import train_test_split
import shutil
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

print('Reading inpainted images - ')

x = []
y = []
globe = glob('Images_inpaint_resized/**/')
for folder in globe:
    label = folder.split('\\')[1]
    for img_path in glob(folder+'*.jpg'):
        x.append(cv2.imread(img_path))
        y.append(label)

num_classes=len(globe)
X=np.array(x)
Y=np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

Y_train_cat = to_categorical(Y_train, num_classes)
Y_test_cat = to_categorical(Y_test, num_classes)

print("All Features:", X.shape)
print("All Labels:", Y.shape)
print("Training Features:", X_train.shape)
print("Training Labels:", Y_train.shape)
print("Testing Features:", X_test.shape)
print("Testing Labels:", Y_test.shape)

directory = 'Images_numpy'
if os.path.exists(directory):
    shutil.rmtree(directory)
os.mkdir(directory)
saveX_train = directory+'/X_train.npy'
saveY_train = directory+'/Y_train.npy'

saveX_test = directory+'/X_test.npy'
saveY_test = directory+'/Y_test.npy'

saveY_train1 = directory+'/Y_train_1dim.npy'
saveY_test1 = directory+'/Y_test_1dim.npy'

np.save(saveX_train,X_train)
np.save(saveY_train,Y_train)
    
np.save(saveX_test,X_test)
np.save(saveY_test,Y_test)

np.save(saveY_train1,Y_train_cat)
np.save(saveY_test1,Y_test_cat)

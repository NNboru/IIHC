#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from tpot import TPOTClassifier
from pycm import ConfusionMatrix
import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Model
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA


# In[2]:


layers_list = ["block4_conv1"]


# In[3]:
image_signal =['viterbi']

for Layer_Featue in layers_list:
    for gname in image_signal:

        try:
            print('$$$$$$$$$$$$$$$$   ',Layer_Featue,'$$$$$$$$$$$$   ',gname)
            default = "/storage/research/tanveer/tuh/data/"
            #train
            directory1 = default+gname+"/train"
            directory2 = default+gname+"/eval"

            X_train = np.load(directory1+'/X_train.npy')
            Y_train = np.load(directory1+'/Y_train.npy')

            X_val = np.load(directory1+'/X_val.npy')
            Y_val = np.load(directory1+'/Y_val.npy')

            Y_train1 = np.load(directory1+'/Y_train_1dim.npy')
            Y_val1 = np.load(directory1+'/Y_val_1dim.npy')

            Y_test1 = np.load(directory2+'/Y_test_1dim.npy')
            X_test = np.load(directory2+'/X_test.npy')
            Y_test = np.load(directory2+'/Y_test.npy')



            print("Training Features:", X_train.shape)
            print("Training Labels:", Y_train.shape)
            print("Test Features:", X_test.shape)

            print("Validation Features:", X_val.shape)
            print("Validation Labels:", Y_val.shape)
            print("Validation Labels 1 dim:", Y_val1.shape)
            print("Validation Labels 1 dim:", Y_train1.shape)
            n_length, n_features, n_outputs = X_train.shape[0], X_train.shape[2], Y_train.shape[1]
            print(X_train.shape[0])
            
            
            ################### Model + Classifier ######################



            X_train_temp = np.concatenate((X_train,X_val), axis=0)
            Y_train_temp = np.concatenate((Y_train,Y_val), axis=0)
            Y_train1_temp = np.concatenate((Y_train1,Y_val1), axis=0)

            X_train = X_train_temp
            Y_train = Y_train_temp
            Y_train1 = Y_train1_temp


            # Edit here Anuragh



            # Model
            print("Model")

            from keras.applications.vgg19 import VGG19 #downloading model for transfer learning
            model_vgg19 = VGG19(include_top=False,weights='imagenet',input_shape=(256,256,3),classes=2)
            optimizer = Adam(lr=0.0001)

            arg_model= Model(inputs=model_vgg19.input, outputs=model_vgg19.get_layer(Layer_Featue).output)
            arg_model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])

            from keras.applications.vgg19 import preprocess_input #preprocessing the input so that it could work with the downloaded model
            bottleneck_train1=arg_model.predict(preprocess_input(X_train),batch_size=50,verbose=1) #calculating bottleneck features, this inshure that we hold the weights of bottom layers
            #bottleneck_val1=arg_model.predict(preprocess_input(X_val),batch_size=50,verbose=1)
            bottleneck_test1=arg_model.predict(preprocess_input(X_test),batch_size=50,verbose=1)


            print(bottleneck_test1.shape)
            print(bottleneck_train1.shape)
            #print(bottleneck_val1.shape)


            train1,train2,train3,train4 = bottleneck_train1.shape
            trainshape = train2*train3*train4
            print('train shape',trainshape)

            test1,test2,test3,test4 = bottleneck_test1.shape
            testshape = test2*test3*test4
            print('test shape',testshape)

            bottleneck_train=bottleneck_train1.reshape(train1,trainshape) 
            #bottleneck_val=bottleneck_val1.reshape(6,featureshape)
            bottleneck_test=bottleneck_test1.reshape(test1,trainshape)

            #bottleneck_train=bottleneck_train1.flatten() 
            #bottleneck_val=bottleneck_val1.flatten()
            #bottleneck_test=bottleneck_test1.flatten()



            print(bottleneck_test.shape)
            print(bottleneck_train.shape)
            saveX_train = directory1+'/bottleneck_train'+Layer_Featue+'.npy'
            saveX_test = directory2+'/bottleneck_test'+Layer_Featue+'.npy'
            
            np.save(saveX_train,bottleneck_train)
            np.save(saveX_test,bottleneck_test)


            print('###### Random Forest #########')
            bottleneck_train = np.load(saveX_train)
            bottleneck_test = np.load(saveX_test)

            from sklearn.ensemble import RandomForestClassifier

            # Paramters setting for the model
            classifer_rf = RandomForestClassifier(n_estimators=500)

            #Train the model
            classifer_rf.fit(bottleneck_train, Y_train1)

            #Predict the values on test data
            logistic_regression_pred = classifer_rf.predict(bottleneck_test)


            from sklearn.metrics import confusion_matrix
            import pandas as pd

            print(pd.DataFrame(confusion_matrix(Y_test1, logistic_regression_pred), index = ['0', '1'], columns = ['0', '1']))

            # Accuracy

            from sklearn.metrics import accuracy_score

            print(accuracy_score(Y_test1, logistic_regression_pred))


            #print(Y_train)
            #arg_model.summary()
            cm1 = confusion_matrix(Y_test1, logistic_regression_pred)
            print('Confusion Matrix : \n', cm1)

            total1=sum(sum(cm1))

            #####from confusion matrix calculate accuracy
            accuracy1=(cm1[0,0]+cm1[1,1])/total1
            print ('Accuracy : ', accuracy1)

            sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            print('Sensitivity : ', sensitivity1 )

            specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
            print('Specificity : ', specificity1)

            from sklearn.metrics import roc_curve, auc,roc_auc_score
            rf_probs = classifer_rf.predict_proba(bottleneck_test)[:, 1]
            #print(rf_probs)
            #print(rf_probs.shape)

            np.save('./rf_probs_'+gname+''+Layer_Featue+'.npy',rf_probs)
            # Calculate roc auc
            roc_value = roc_auc_score(Y_test1, rf_probs)
            print('ROC Value = ',roc_value )
            np.save('./rf_'+gname+''+Layer_Featue+'.npy',logistic_regression_pred)


            base_fpr, base_tpr, _ = roc_curve(Y_test1, [1 for _ in range(len(Y_test1))])
            model_fpr, model_tpr, _ = roc_curve(Y_test1, rf_probs)

            plt.figure(figsize = (8, 6))
            plt.rcParams['font.size'] = 16
                
                # Plot both curves
            plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
            plt.plot(model_fpr, model_tpr, 'r', label=Layer_Featue+'(area = %0.2f)' % roc_value)
            plt.legend();
            plt.xlabel('False Positive Rate'); 
            plt.ylabel('True Positive Rate'); 
            plt.title('ROC Curves');
            plt.savefig('./roc_'+gname+''+Layer_Featue+'.png')
            plt.close

            # Classification Report

            from sklearn.metrics import classification_report
            print(classification_report(Y_test1, logistic_regression_pred))

            from pycm import ConfusionMatrix
            cm = ConfusionMatrix(actual_vector=list(Y_test1),predict_vector=list(logistic_regression_pred))
            print(cm)
            print('Geometric means')
            from imblearn.metrics import geometric_mean_score
            print(geometric_mean_score(Y_test1, logistic_regression_pred, average='weighted'))
            print('End Geometric means')
            tpot = TPOTClassifier(generations=5,verbosity=2)
            tpot.fit(bottleneck_train, Y_train1)
            print(tpot.score(bottleneck_test, Y_test1))
            logistic_regression_pred = tpot.predict(bottleneck_test)
            
            cm = ConfusionMatrix(actual_vector=list(Y_test1),predict_vector=list(logistic_regression_pred))
            print(cm)


            # In[ ]:


            print('$$$$$$$$$$$$$$$$   ',Layer_Featue,'   $$$$$$$$$$$$   ',gname)

        except Exception as e:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@  errror@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(e)


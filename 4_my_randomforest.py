from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

# Random Forest

layers_list = ["block4_conv1"]

for Layer_Feature in layers_list:
    try:
        print('\n$$$$$$$$$$$$  ',Layer_Feature,'  $$$$$$$$$$$$\n')
        
        print('###### Random Forest #########')

        new_folder = 'numpy_bottleneck_data'
        saveX_train = new_folder+'/train'+Layer_Feature+'.npy'
        saveX_test = new_folder+'/test'+Layer_Feature+'.npy'
        bottleneck_train = np.load(saveX_train)
        bottleneck_test = np.load(saveX_test)

        folder  = 'numpy_data/'
        Y_train = np.load(folder + 'Y_train.npy')
        Y_test = np.load(folder + 'Y_test.npy')


        # Paramters setting for the model
        classifer_rf = RandomForestClassifier(n_estimators=500)

        #Train the model
        classifer_rf.fit(bottleneck_train, Y_train)

        #Predict the values on test data
        logistic_regression_pred = classifer_rf.predict(bottleneck_test)



        # Accuracy
        
        print(pd.DataFrame(confusion_matrix(Y_test, logistic_regression_pred), index = ['0', '1'], columns = ['0', '1']))

        print(accuracy_score(Y_test, logistic_regression_pred))


        #arg_model.summary()
        cm1 = confusion_matrix(Y_test, logistic_regression_pred)
        print('Confusion Matrix : \n', cm1)

        total1=sum(cm1)

        #####from confusion matrix calculate accuracy
        accuracy1=(cm1[0,0]+cm1[1,1])/total1
        print ('Accuracy : ', accuracy1)

        sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        print('Sensitivity : ', sensitivity1 )

        specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        print('Specificity : ', specificity1)


    except Exception as e:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@  errror  @@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(e)
            if not int(input('continue ( 0:no, 1:yes ) : ')) :
                break

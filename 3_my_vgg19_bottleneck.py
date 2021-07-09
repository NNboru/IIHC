from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19

layers_list = ["block4_conv1"]

for Layer_Feature in layers_list:
    try:
        print('\n$$$$$$$$$$$$  ',Layer_Feature,'  $$$$$$$$$$$$\n')
        
        #train
        folder  = 'Images_numpy/'
        X_train = np.load(folder + 'X_train.npy')
        X_test = np.load(folder + 'X_test.npy')

        print("Training Features:", X_train.shape)
        print("Test Features:", X_test.shape)

        # n_length, n_features, n_outputs = X_train.shape[0], X_train.shape[2], Y_train.shape[0]



        ################### Model + Classifier ######################

        # downloading model for transfer learning
        # from keras.applications.vgg19 import VGG19

        # init model
        print('Initializing VGG19 -')
        model_vgg19 = VGG19(include_top=False,weights='imagenet',input_shape=(256,256,3),classes=136)
        optimizer = Adam(lr=0.0001)

        arg_model= Model(inputs=model_vgg19.input, outputs=model_vgg19.get_layer(Layer_Featue).output)
        arg_model.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])

        #preprocessing the input so that it could work with the downloaded model
        #calculating bottleneck features, this insure that we hold the weights of bottom layers
        bottleneck_train1=arg_model.predict(preprocess_input(X_train),batch_size=50,verbose=1)
        bottleneck_test1=arg_model.predict(preprocess_input(X_test),batch_size=50,verbose=1)


        print('bottleneck_train1:', bottleneck_train1.shape)
        print('bottleneck_test1:' , bottleneck_test1.shape)


        train1,train2,train3,train4 = bottleneck_train1.shape
        trainshape = train2*train3*train4
        print('train shape',trainshape)

        test1,test2,test3,test4 = bottleneck_test1.shape
        testshape = test2*test3*test4
        print('test shape',testshape)

        bottleneck_train=bottleneck_train1.reshape(train1,trainshape)
        bottleneck_test=bottleneck_test1.reshape(test1,trainshape)

        #bottleneck_train=bottleneck_train1.flatten() 
        #bottleneck_val=bottleneck_val1.flatten()
        #bottleneck_test=bottleneck_test1.flatten()

        new_folder = 'numpy_bottleneck_data'
        saveX_train = new_folder+'/train'+Layer_Feature+'.npy'
        saveX_test = new_folder+'/test'+Layer_Feature+'.npy'
        
        np.save(saveX_train,bottleneck_train)
        np.save(saveX_test,bottleneck_test)


    except Exception as e:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@  errror  @@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
            print(e,'\n')
            if Layer_Feature!=layers_list[-1] and not int(input('continue ( 0:no, 1:yes ) : ')) :
                break


        

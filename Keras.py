import numpy as np
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot_ng as pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import kt_utils
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def Happymodel(input_shape):
    x_input=Input(input_shape)
    x=ZeroPadding2D((3,3))(x_input)
    x=Conv2D(32,(3,3),strides=(1,1),name='conv0')(x)
    x=BatchNormalization(axis=3,name='bn0')(x)
    x=Activation('relu')(x)
    x=MaxPooling2D((2,2),name='max_pool')(x)
    x=Flatten()(x)
    x=Dense(1,activation='sigmoid',name='fc')(x)

    model=Model(inputs=x_input,outputs=x,name='Happymodel')
    return model

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T



happy_model=Happymodel(X_train.shape[1:])


happy_model.compile('adam','binary_crossentropy',metrics=['accuracy'])
happy_model.fit(X_train,Y_train,epochs=40,batch_size=50)
preds=happy_model.evaluate(X_test,Y_test,batch_size=32,verbose=1,sample_weight=None)
print ("误差值 = " + str(preds[0]))
print ("准确度 = " + str(preds[1]))






















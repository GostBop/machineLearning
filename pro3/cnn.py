import numpy as np
import os,random,shutil

import pandas as pd
np.random.seed(7)
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.applications import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2

train_data_dir=os.path.join('../input/image-cnn-2/train/train')
print(train_data_dir)
val_data_dir=os.path.join('../input/image-cnn-2/test_/test_')
test_data_dir=os.path.join('../input/image-cnn-2/test/test')

IMG_W,IMG_H,IMG_CH=299,299,3 # 单张图片的大小
batch_size=60
epochs=50  
class_num=8


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( # 单张图片的处理方式，train时一般都会进行图片增强
        rescale=1. / 255, # 图片像素值为0-255，此处都乘以1/255，调整到0-1之间
        shear_range=0.2, # 斜切
        zoom_range=0.2, # 放大缩小范围
        horizontal_flip=True
        ) # 水平翻转

train_generator = train_datagen.flow_from_directory(# 从文件夹中产生数据流
    train_data_dir, # 训练集图片的文件夹
    target_size=(IMG_W, IMG_H), # 调整后每张图片的大小
    batch_size=batch_size,
    class_mode='categorical') # 此处是多分类问题，故而mode是categorical

# 3，同样的方式准备测试集
val_datagen = ImageDataGenerator(rescale=1. / 255) # 只需要和trainset同样的scale即可，不需增强
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical')

resnet_weights_path = '../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
# 4，建立Keras模型：模型的建立主要包括模型的搭建，模型的配置
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.optimizers import SGD
from keras import regularizers
from keras.constraints import max_norm
def build_model(input_shape):

    input_tensor = Input(shape=(299, 299, 3))
    # 构建不带分类器的预训练模型
    base_model = InceptionResNetV2(weights=resnet_weights_path,input_tensor=input_tensor, include_top=False)
    
    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    '''
    #x = Dropout(0.5)(x)
    def l1_reg(weight_matrix):
        return 0.01 * K.sum(K.abs(weight_matrix))
    # 添加一个全连接层
    x = Dense(1024, activation='relu',                 
                kernel_initializer='random_uniform',
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l1_l2(1e-4),
                #activity_regularizer=regularizers.l1(0.01),
                kernel_constraint=max_norm(2.)
                )(x)
    x = Dropout(0.5)(x)'''
    predictions = Dense(8, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    learning_rate = 0.1
    decay = 0.001
    sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model # 返回构建好的模型


model=build_model(input_shape=(IMG_W,IMG_H,IMG_CH))

history_ft = model.fit_generator(train_generator,
                        steps_per_epoch=100,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=50,
                        callbacks=[EarlyStopping(monitor="val_acc", patience=5)])


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_data_dir, 
                                target_size=(IMG_W, IMG_H),
                                batch_size=1,
                                class_mode='categorical', 
                                shuffle=False,)


#预测
test_generator.reset()
pred = model.predict_generator(test_generator, verbose=1,
                        steps=4500)

predicted_class_indices = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)
label = dict((v,k) for k,v in labels.items())

predictions = [label[i] for i in predicted_class_indices]

filenames = test_generator.filenames

dataframe = pd.DataFrame({'Image':filenames,'Cloth_label':predictions})
dataframe.to_csv("result.csv",index=False,sep=',')
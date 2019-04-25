import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, Flatten, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam
from keras.utils import multi_gpu_model


DATASET_DIR = '/mnt/wd2T/wcz/ImageNet_Data/raw-data/train'

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


base_model=keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha = 0.75,depth_multiplier = 1, dropout = 0.001, pooling='avg',include_top = False, weights = "imagenet", classes = 1000)
x=base_model.output
x = Dropout(0.001, name='dropout')(x)  #drop=0.001
preds=Dense(1000,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

for i,layer in enumerate(model.layers):
    print(i,layer.name)

for layer in model.layers[:86]:
    layer.trainable=False
for layer in model.layers[86:]:
    layer.trainable=True

paralleled_model=multi_gpu_model(model, gpus=2)
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 
train_generator=train_datagen.flow_from_directory(DATASET_DIR,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=512,
                                                 class_mode='categorical', shuffle=True)
paralleled_model.summary()
paralleled_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy


# checkpoint
class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('model_at_epoch_%d.h5' % epoch)

checkpoint = MyCbk(model)
callbacks_list = [checkpoint]

# 50step/epoch, 20epoch, about 4 hours in dual-1080Ti, accuray ~60%
step_size_train=50  #train_generator.n//train_generator.batch_size/8
paralleled_model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,callbacks=callbacks_list,epochs=20)

model.save('mbnet75.h5')

#model.load_weights('mbnet75.h5')
#preprocessed_image = load_image('test.jpg')
#predictions = model.predict(preprocessed_image)
#print(predictions[0][xxx]*100)






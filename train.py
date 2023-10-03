import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
def get_video_labels( path_subset , mode = 'label'):
    if mode == 'input':
        names = ["video_id"]
    elif mode == 'label':
        names = ["video_id","label"]
    df = pd.read_csv(path_subset, sep=";" ,names = names)                                       
    if mode == 'label':
        df = df[df.label.isin(labels)]
    return df
                         
data_root = "/home/arneshbatra2212/Desktop/AI/GestureControl/V2"
csv_labels = "/home/arneshbatra2212/Desktop/AI/GestureControl/V2/labels.csv"
csv_train = "/home/arneshbatra2212/Desktop/AI/GestureControl/V2/train.csv"
csv_test = "/home/arneshbatra2212/Desktop/AI/GestureControl/V2/test.csv"
csv_val = "/home/arneshbatra2212/Desktop/AI/GestureControl/V2/validation.csv"
data_vid = "/home/arneshbatra2212/Desktop/DATA/gesture_data/20bn-jester-v1"


labels_df = pd.read_csv(csv_labels , header = None)
labels = [str(label[0]) for label in labels_df.values]
n_labels = len(labels)
#create dictionary
labels_to_int = dict(zip(labels,range(n_labels)))
int_to_labels = dict(zip(range(n_labels),labels))



train_df = get_video_labels(csv_train)
test_df = get_video_labels(csv_test, mode = 'input')
val_df = get_video_labels(csv_val )

model_name = "resnet_3d_model"
target_size = (64,95)
nb_frames = 16
skip = 1
batch_size = 32
input_shape = (nb_frames,) + target_size + (3,)


print(input_shape)

import tensorflow as tf
from image import ImageDataGenerator
datagen=ImageDataGenerator(rescale = 1./255)

gen_train = datagen.flow_video_from_dataframe(
    train_df,
    directory=data_vid,
    x_col='video_id',
    y_col='label',
    target_size=target_size,
    path_classes=csv_labels,

    batch_size=batch_size,
    nb_frames=nb_frames,
    skip=skip,
    has_ext = True
    
    
)

gen_val = datagen.flow_video_from_dataframe(
    val_df,
    directory=data_vid,
    x_col='video_id',
    y_col='label',
    target_size=target_size,
    path_classes=csv_labels,

    batch_size=batch_size,
    nb_frames=nb_frames,
    skip=skip,
    has_ext = True
    
    
)

from resnet_model import Resnet3DBuilder

from tensorflow.keras.optimizers.legacy import SGD
nb_classes = 27
resnet_model = Resnet3DBuilder.build_resnet_101(input_shape , nb_classes , drop_rate=0.5) 
optimizer = SGD(learning_rate=0.0001, momentum=0.9, nesterov=False)
resnet_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model_file = "/home/arneshbatra2212/Desktop/AI/GestureControl/V2/resnet_3d_model/resnetmodel.hdf5"


from tensorflow.keras.callbacks import ModelCheckpoint

model_checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


#callbacks and graphs
import datetime

csv_logger = tf.keras.callbacks.CSVLogger("training.log", separator=",", append=False)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



from math import ceil

resnet_model.fit(
    gen_train, steps_per_epoch=ceil(len(train_df.video_id)/batch_size), epochs=100, validation_data=gen_val,
    validation_steps=30 ,shuffle=True, workers = 8 ,use_multiprocessing= False , max_queue_size=20,
    verbose=1, callbacks=[model_checkpoint, tensorboard_callback , csv_logger ]
)

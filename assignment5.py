from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.data_utils import to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data
from tflearn.layers.core import dropout
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import max_pool_2d
from tflearn.layers.estimator import regression

#to download data from the script which is present in tflearn
from tflearn.datasets import cifar10
(in_train, ground_train), (in_test, ground_test) = cifar10.load_data()

#preprocessing the input data
#shuffling the data
in_train, ground_train = shuffle(in_train, ground_train)

#saying that the number of labels used are 10
ground_train = to_categorical(ground_train, 10)
ground_test = to_categorical(ground_test, 10)

preprocess_data = ImagePreprocessing()
preprocess_data.add_featurewise_zero_center()
preprocess_data.add_featurewise_stdnorm()

#augmentation
augment_data = ImageAugmentation()
augment_data.add_random_flip_leftright()
augment_data.add_random_rotation(max_angle=25.0)
augment_data.add_random_blur(sigma_max = 3.0)

#network
net_var = input_data(shape = [None, 32, 32, 3], data_preprocessing=preprocess_data, data_augmentation=augment_data)
net_var = conv_2d(net_var, 32, 3, activation='relu')
net_var = max_pool_2d(net_var, 2)
net_var = conv_2d(net_var, 64, 3, activation='relu')
net_var = conv_2d(net_var, 64, 3, activation='relu')
net_var = max_pool_2d(net_var, 2)
net_var = fully_connected(net_var, 512, activation='relu')
#net_var = dropout(net_var, 0.5)
net_var = fully_connected(net_var, 10, activation='softmax')

sgd = tflearn.optimizers.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=100)
top_k = tflearn.metrics.Top_k(3)
#net_var = regression(net_var, optimizer=sgd, loss='categorical_crossentropy')
net_var = regression(net_var, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

#train
model = tflearn.DNN(net_var, tensorboard_verbose=0)
model.fit(in_train, ground_train, n_epoch=100, shuffle=True, validation_set=(), show_metric=True, batch_size=100, run_id='assginment5_adam_with_dropout')

import tensorflow as tf

import numpy as np

from tensorflow import keras

list_1=[5,10,15,20,25]
list_2=[1,2,3,4,5]

model= keras.Sequential([keras.layers.Dense(units= 1, input_shape= [1])])

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

history= model.fit(list_2, list_1, epochs=1000)

print(model.predict([10]))
import os 
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
import cv2
from PIL import Image
from keras.datasets import mnist
import matplotlib.pyplot as plt     

(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = load_model('my_model.h5')

digits = ['0','1','2','3','4','5','6','7','8','9']
x_test = np.expand_dims(x_test, axis=3)
predict = model.predict(x_test) 

for i in range(20):      
    plt.subplot(5,4,i+1)     
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i].reshape(28,28),cmap = 'binary')        
    plt.title('Result:'+ digits[np.argmax(predict[i])])
plt.show()
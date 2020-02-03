import os 
import numpy as np
import keras
from keras.models import load_model
import cv2
from PIL import Image

model = load_model('my_model.h5')
digits = ['0','1','2','3','4','5','6','7','8','9']

cap = cv2.VideoCapture(0)

while(True):  
    _, rgb = cap.read()  
    frame  = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)/255.0
    # my webcam size 480x640
    im = Image.fromarray(frame[0:480,80:560])
    im = im.resize((28,28))
    img_array = np.array(im)
    # tensor shape : 1x28x28x1
    img_array = np.expand_dims(img_array, axis=2)
    img_array = np.expand_dims(img_array, axis=0)    

    predict = model.predict(1.0-img_array)    
    result =  digits[np.argmax(predict)]    
    print('Result:', result)
    cv2.putText(img=rgb,text=result,org=(240,320),fontFace=cv2.FONT_HERSHEY_DUPLEX,color=(0, 255, 255),fontScale=9)
    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



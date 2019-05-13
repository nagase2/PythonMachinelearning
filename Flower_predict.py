from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

model = model_from_json(open('flower_predict.json').read())

model.load_weights('flower_predict.hdf5')

categories = ["daisy","dandelion","roses","sunflowers","tulips"]

img_path = "tulip2.jpg"
img = image.load_img(img_path,target_size=(300, 300, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

features = model.predict(x)

print(features)
if features[0,0] == 1:
    print ("aaaaaa")

elif features[0,4] == 1:
    print ("dddddd")

else:
   
   
    message = "xxxxxxx"
    print(message)
    
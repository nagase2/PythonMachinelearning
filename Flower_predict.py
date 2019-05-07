from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

model = model_from_json(open('flower_predict.json').read())

model.load_weights('flower_predict.hdf5')

categories = ["daisy","dandelion","roses","sunflowers","tulips"]

img_path = str(input())
img = image.load_img(img_path,target_size=(300, 300, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

features = model.predict(x)

if features[0,0] == 1:
    print ("選ばれたのは、綾鷹でした。")

elif features[0,4] == 1:
    print ("選ばれたのは、綾鷹（茶葉のあまみ）でした。")

else:
    for i in range(0,10):
          if features[0,i] == 1:
              cat = categories[i]
    message = "綾鷹を選んでください。（もしかして：あなたが選んでいるのは「" + cat + "」ではありませんか？）"
    print(message)
    
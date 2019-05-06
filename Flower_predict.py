
from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('保存先のフォルダ/tea_predict.json').read())
#保存した重みの読み込み
model.load_weights('保存先のフォルダ/tea_predict.hdf5')

categories = ["綾鷹","お〜いお茶　抹茶入り","なごみ","お〜いお茶　新茶","綾鷹　茶葉のあまみ",
              "お〜いお茶","伊右衛門","お〜いお茶　濃い茶","生茶","お〜いお茶　新緑"]

#画像を読み込む
img_path = str(input())
img = image.load_img(img_path,target_size=(250, 250, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

#予測結果によって処理を分ける
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
    
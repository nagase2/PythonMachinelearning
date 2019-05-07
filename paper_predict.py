# predict data from input using trained model

from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

# 保存したモデルの読み込み
model = model_from_json(open('paper_predict.json').read())
# 保存した重みの読み込み
model.load_weights('paper_predict.hdf5')

categories = ["paper_bent", "paper_clean"]

# 画像を読み込む
img_path = "paperpic.jpg"
img = image.load_img(img_path, target_size=(300, 300, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 予測
features = model.predict(x)

print(features[0,0], features[0,1])

# 予測結果によって処理を分けるd
if features[0, 0] > features[0, 1] :
    print("曲がった髪が選ばれました")
else:
    print("きれいな紙が選ばれました")


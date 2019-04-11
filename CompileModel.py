from keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(250,250,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(3,activation="sigmoid")) #分類先の種類分設定

#モデル構成の確認
model.summary()

#モデルのコンパイル

from keras import optimizers

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])


#データの準備

from keras.utils import np_utils
import numpy as np



# 商品名
categories = ["bita","irohasu","milk"]


nb_classes = len(categories)

train_data = np.load("tea_data2.npz")

X_train = train_data['X_train']
X_test = train_data['X_test']
y_train = train_data['y_train']
y_test = train_data['y_test']

#データの正規化
X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float")  / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)


#モデルの学習

model = model.fit(X_train,
                  y_train,
                  epochs=20,
                  batch_size=6,
                  validation_data=(X_test,y_test))


#学習結果を表示

import matplotlib.pyplot as plt

acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('精度を示すグラフのファイル名')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('損失値を示すグラフのファイル名')




#モデルの保存

json_string = model.model.to_json()
open('tea_predict.json', 'w').write(json_string)

#重みの保存

hdf5_file = "tea_predict.hdf5"
model.model.save_weights(hdf5_file)


#ラベリングによる学習/検証データの準備

from PIL import Image
import os, glob
import numpy as np
import random, math

#画像が保存されているルートディレクトリのパス
root_dir = "img"
# 商品名
categories = ["bita","milk","irohasu"]



#
# # 画像データ用配列
# X = []
# # ラベルデータ用配列
# Y = []


#全データ格納用配列
allfiles = []

#カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for idx, cat in enumerate(categories):
    basewidth = 300
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpeg")
    for f in files:
        img = Image.open(f)
        #img.show()
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        #同じファイル名で保存する
        img.save(f.title())







# basewidth = 100
# img = Image.open('./new.jpg')
# print(img.size)
# wpercent = (basewidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((basewidth,hsize), Image.ANTIALIAS)
# img.save('new.jpg')
import os
import numpy as np
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 猫の種類リスト
kind_list = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
             'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
             'Siamese','Sphynx']
img_size = 224
img_cat = []
kind_label = []

# 画像データの読み込みと前処理
#　３００だったが仮に１８０に変更
for i in kind_list:
    for j in range(1, 180):
        img_path = f'C:/Users/tenai/Desktop/猫AI/images/{i}_{j}.jpg'
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(img_size, img_size))
            x = img_to_array(img)
            img_cat.append(x)
            kind_label.append(kind_list.index(i))

X = np.array(img_cat)
X = X / 255.0  # 画像を正規化
y = to_categorical(kind_label, num_classes=len(kind_list))

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# モデルの構築
input_tensor = Input(shape=(img_size, img_size, 3))
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='sigmoid'))
top_model.add(Dense(len(kind_list), activation='softmax'))

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# 転移学習のため、ベースモデルの重みを固定
for layer in base_model.layers:
    layer.trainable = False

# モデルのコンパイル
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=1e-4, momentum=0.9), metrics=['accuracy'])

# モデルの学習
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# モデルの保存
model.save_weights('model_weight.weights.h5')

# 学習履歴の可視化
plt.plot(history.history['accuracy'], label="acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.show()

# 精度の評価
scores = model.evaluate(X_test, y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
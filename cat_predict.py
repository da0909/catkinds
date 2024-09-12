import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

# 猫の種類リスト
classes = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
           'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
           'Siamese','Sphynx']
image_size = 224

# モデルの構築（重みのみ読み込み）
def create_model():
    input_tensor = Input(shape=(image_size, image_size, 3))
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='sigmoid'))
    top_model.add(Dense(len(classes), activation='softmax'))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # コンパイル
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('model_weight.weights.h5')  # 保存された重みをロード
    return model

# 猫の種類を予測する関数
def predict_cat_breed(img_path, model):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    predicted_percentage = prediction[0][predicted_index] * 100

    return predicted_class, predicted_percentage

# メイン処理
def main():
    model = create_model()
    # img_path = input("画像ファイルのパスを入力してください: ")
    img_path = R"C:\Users\tenai\Desktop\猫AI\images\Bombay_157.jpg"
    
    if os.path.exists(img_path):
        predicted_class, predicted_percentage = predict_cat_breed(img_path, model)
        print(f"この猫は {predicted_class} です（確率: {predicted_percentage:.2f}％）")
    else:
        print("指定されたファイルが見つかりません。")

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr

tmp = []
for i in range(1,10):
  urlretrieve(f"https://github.com/MaxWutw/Deep-Learning/raw/main/Image%20recognition/photo{i}.jpg", f"photo{i}.jpg")
  tmp.append(f"photo{i}.jpg")

data = []
for i in range(0,9):
  img = load_img(tmp[i], target_size = (256,256))
  x = img_to_array(img)
  data.append(x)
data = np.asarray(data, dtype = np.uint8)

target = np.array([1,1,1,2,2,2,3,3,3])

x_train = preprocess_input(data)
plt.axis('off')
n = 1
plt.imshow(x_train[n])

y_train = to_categorical(target-1, 3)
y_train[n]

resnet = ResNet50(include_top=False, pooling="avg")
resnet.trainable = False

model = Sequential()
model.add(resnet)
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=9, epochs=25)

y_predict = np.argmax(model.predict(x_train), -1)
labels=["麻雀", "喜鵲", "白頭翁"]
print(y_predict)
print(target-1)

testing = []
pho = []
for i in range(1,4):
  urlretrieve(f"https://github.com/MaxWutw/Deep-Learning/raw/main/Image%20recognition/test{i}.jpg", f"test{i}.jpg")
  pho.append(f"test{i}.jpg")
for i in range(0,3):
  img = load_img(pho[i], target_size = (256,256))
  x = img_to_array(img)
  testing.append(x)
testing = np.asarray(testing, dtype = np.uint8)

testing_data = preprocess_input(testing)
final = np.argmax(model.predict(testing_data), -1)
for i in range(3):
  print(f"{i+1}. CNN judge: ", labels[final[i]])

print("Answer : 麻雀、白頭翁、喜鵲")

def classify_image(inp):
  inp = inp.reshape((-1, 256, 256, 3))
  inp = preprocess_input(inp)
  prediction = model.predict(inp).flatten()
  return {labels[i]: float(prediction[i]) for i in range(3)}

image = gr.inputs.Image(shape=(256, 256), label="鳥類照片")
label = gr.outputs.Label(num_top_classes=3, label="AI辨識結果")

gr.Interface(fn=classify_image, inputs=image, outputs=label,
             title="AI 三種鳥類辨識機",
             description="我能辨識台灣常見的三種鳥類: 麻雀、喜鵲、白頭翁。",
             capture_session=True).launch()

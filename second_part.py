import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from io import BytesIO
from PIL import Image

from labels import labels

model = keras.applications.VGG16()
plt.figure()
for i in range(1, 4):
  uploaded = None
  with open(f'image{i}.jpg', 'rb') as file:
    uploaded = file.read()
  img = Image.open(BytesIO(uploaded))
  ax = plt.subplot(1, 3, i)
  plt.imshow( img )
  # приводим к входному формату VGG-сети
  img = np.array(img)
  x = keras.applications.vgg16.preprocess_input(img)
  print(x.shape)
  x = np.expand_dims(x, axis=0)
  # прогоняем через сеть
  res = model.predict( x )
  idx = np.argmax(res)
  print(idx)
  label = labels[idx]
  ax.set_title(label)
  print(label)
plt.show()

#%%
from typing import Callable
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, InputLayer
from keras import utils
from keras.optimizers import Adam
import numpy as np
from colorama import Fore, Style
import time
from functools import wraps

def timeit(func: Callable):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return (end-start), result
  return wrapper

LEARNING_RATE = 0.01
BATCH_SIZE = 60
EPOCHS = 5

def create_model(learning_rate: float, conv_layers: list, dense_layers: list) -> Sequential:
  model = Sequential([
      InputLayer(input_shape=(28, 28, 1)),
      *(
        layer(filters, ks, ps) for (filters, ks, ps) in conv_layers for layer in (
          lambda filters, ks, _: Conv2D(filters=filters, kernel_size=ks, padding='same', activation='relu'),
          lambda _1, _2, ps: MaxPooling2D(pool_size=ps, strides=2)
          )
      ),
      Flatten(),
      *(Dense(n, activation='relu') for n in dense_layers),
      Dense(10, activation='softmax')
  ])

  model.compile(optimizer=Adam(learning_rate),
                loss='categorical_crossentropy', metrics=['accuracy'])
  return model


def train_model(model: Sequential, x: np.ndarray, y: np.ndarray, epochs: int):
  model.fit(x, y, batch_size=BATCH_SIZE,
            epochs=epochs, validation_split=0.1, verbose=0)

def resolve_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = np.expand_dims(x_train / 255, axis=3)
  x_test = np.expand_dims(x_test / 255, axis=3)
  y_train_cat = utils.to_categorical(y_train, 10)
  y_test_cat = utils.to_categorical(y_test, 10)
  return x_train, x_test, y_train_cat, y_test_cat

@timeit
def train_and_eval(model: Sequential, dataset: tuple[int], epochs: int) -> float:
  x_train, x_test, y_train, y_test = dataset
  train_model(model, x_train, y_train, epochs)
  return model.evaluate(x_test, y_test, verbose=0, return_dict=True)['accuracy']

def test_performance():
  dataset = resolve_dataset()
  results = []
  configs = [
    {"conv": [(32, (3, 3), (2, 2)), (64, (3, 3), (2, 2))], "dense": [128], "epochs": 5, "learning_rate": 0.01},
    {"conv": [(40, (5, 5), (2, 2)), (100, (3, 3), (2, 2))], "dense": [140], "epochs": 5, "learning_rate": 0.01},
    {"conv": [(40, (5, 5), (2, 2)), (100, (3, 3), (2, 2))], "dense": [64], "epochs": 5, "learning_rate": 0.01},
    {"conv": [(40, (5, 5), (2, 2)), (100, (3, 3), (2, 2))], "dense": [64], "epochs": 10, "learning_rate": 0.01},
    {"conv": [(40, (5, 5), (2, 2)), (100, (3, 3), (2, 2))], "dense": [64], "epochs": 10, "learning_rate": 0.1},
    ]
  for config in configs:
      model = create_model(config['learning_rate'], config['conv'], config['dense'])
      time_spent, accuracy = train_and_eval(model, dataset, config['epochs'])
      print(f'{time_spent=}s, {accuracy=}')
      results.append((config, accuracy, time_spent))

  print(results)

#%% 
def example_run():
  x_train, x_test, y_train, y_test = resolve_dataset()

  plt.figure(figsize=(10, 5))
  for i in range(25):
      plt.subplot(5, 5, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(x_train[i], cmap="binary")
  plt.show()

  model: Sequential = create_model(LEARNING_RATE)
  print(Fore.GREEN, "Model summary:", Style.RESET_ALL)
  model.summary()

  print(Fore.GREEN, "Training the model", Style.RESET_ALL)
  BATCH_SIZE = 60
  EPOCHS = 25
  model.fit(x_train, y_train, batch_size=BATCH_SIZE,
            epochs=EPOCHS, validation_split=0.1)

  print(Fore.GREEN, "Model prediction occuracy", Style.RESET_ALL)
  model.evaluate(x_test, y_test)

  n = 1
  x = np.expand_dims(x_test[n], axis=0)
  res = model.predict(x)
  print(Fore.GREEN,
        f"Evaluation on sample #{n+1} (certainties)", Style.RESET_ALL)
  print([f"{int(100*r)}%" for r in res[0]])
  print(Fore.GREEN, "Evaluation result", Style.RESET_ALL)
  print(np.argmax(res))
  plt.imshow(x_test[n], cmap="binary")
  plt.show()

  pred_prob = model.predict(x_test)
  pred = np.argmax(pred_prob, axis=1)
  print(Fore.GREEN, "Test set evaluation result shape", Style.RESET_ALL)
  print(pred.shape)
  N = 20
  print(Fore.GREEN, f"{N} first predicted values", Style.RESET_ALL)
  print(pred[:N])
  print(Fore.GREEN, f"{N} first reference values", Style.RESET_ALL)
  print(y_test[:N])

  mask = pred == y_test
  print(Fore.GREEN, f"{N} first truthiness mask values", Style.RESET_ALL)
  print(mask[:N])
  x_false = x_test[~mask]
  y_false = y_test[~mask]
  pred_false = pred[~mask]
  pred_prob_false = pred_prob[~mask]
  print(Fore.GREEN, f"Shape of subset of original test set values on which nn had wrong prediction", Style.RESET_ALL)
  print(x_false.shape)
  N = 5
  print(Fore.GREEN,
        f"{N} first wrong nn predictions compared to reference", Style.RESET_ALL)
  
  plt.figure()
  for i in range(5):
    print(
        f"Predicted value: {pred_false[i]}, Certainty: {pred_prob_false[i][pred_false[i]]*100:.1}%")
    print(f"Reference value: {y_false[i]}")
    plt.subplot(5, 1, i)
    plt.imshow(x_false[i], cmap="binary")
  plt.show()


if __name__ == '__main__':
  test_performance()

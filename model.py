import tensorflow as tf
import keras
from keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

def conv(x , f, k , s= 1):
    x = layers.Conv2D(f, k , s, padding = "same" , use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.1)(x)

def yolov1():
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  base_model.trainable = False

  x = base_model.output

  x = layers.Conv2D(1024,3, padding = "same" , use_bias = False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(0.1)(x)

  x = layers.Flatten()(x)
  x = layers.Dense(4096)(x)
  x = layers.LeakyReLU(0.1)(x)
  x = layers.Dropout(0.5)(x)

  x = layers.Dense(7*7*30, activation = "linear")(x)
  out = layers.Reshape((7,7,30))(x)

  return models.Model(base_model.input, outputs= out)

if __name__ == "__main__":
    model = yolov1()
    model.summary()




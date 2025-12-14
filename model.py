import tensorflow as tf
from keras import layers, models

def conv(x , f, k , s= 1):
    x = layers.Conv2D(f, k , s, padding = "same" , use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.1)(x)

def yolov1():
    inp = layers.Input((448, 448, 3))
    x = conv(inp, 64, 7 , 2)
    x = layers.MaxPooling2D(2)(x)
    x = conv(x, 192, 3)
    x = layers.MaxPooling2D(2)(x)

    x = conv(x, 128,1)
    x = conv(x, 256, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = layers.MaxPooling2D(2)(x)

    for _ in range(4):
        x = conv(x, 256, 1)
        x = conv(x, 512, 3)


    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = layers.MaxPooling2D(2)(x)

    x = conv(x, 1024, 3)
    x = conv(x, 1024, 3)

    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(7*7*30)(x)
    out = layers.Reshape((7,7,30))(x)

    return models.Model(inp, out)

if __name__ == "__main__":
    model = yolov1()
    model.summary()



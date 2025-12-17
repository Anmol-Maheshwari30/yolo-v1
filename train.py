import tensorflow as tf
import numpy as np
from datasets import *
from model import yolov1
from loss import yolo_loss

voc_root = r"data\VOCtrainval_06-Nov-2007\VOCdevkit"
ids = load_image_ids(voc_root)[:100]

def voc_generator (image_ids, voc_root, batch_size = 16):
  while True:
    np.random.shuffle(image_ids)
    for i in range(0, len(image_ids) , batch_size):
      batch_ids = image_ids[i:i+batch_size]
      x_batch,y_batch = [],[]
      for img_id in ids:
        img = load_image(voc_root, img_id)
        if img is None:
          continue
        boxes, labels = parse_annotation(voc_root, img_id)
        target = encode_yolo_target(boxes, labels)
        x_batch.append(img)
        y_batch.append(target)

        if len(x_batch) > 0:
           yield np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

all_ids = load_image_ids(voc_root)
train_gen = voc_generator(all_ids, voc_root, batch_size = 16)

model = yolov1()
model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate = 1e-4), loss = yolo_loss)
model.fit(train_gen, steps_per_epoch = len(all_ids)//16,epochs = 5)

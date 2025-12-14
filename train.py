import tensorflow as tf
import numpy as np
from datasets import *
from model import yolov1
from loss import yolo_loss

voc_root = r"data\VOCtrainval_06-Nov-2007\VOCdevkit"

ids = load_image_ids(voc_root)[:5]

x, y =[],[]
for img_id in ids:
    img = load_image(voc_root, img_id)
    if img is None:
        continue
    boxes, labels = parse_annotation(voc_root, img_id)
    target = encode_yolo_target(boxes, labels)
    x.append(img)
    y.append(target)

x = np.array(x , dtype = np.float32)
y = np.array(y , dtype = np.float32)

model = yolov1()
model.compile(optimizer = "adam" , loss = yolo_loss)
model.fit(x,y,epochs = 5)


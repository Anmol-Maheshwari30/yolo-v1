import cv2
import numpy as np
from model import yolov1
from datasets import load_image_ids, load_image
from utils import decode_predictions, draw_boxes

voc_root = f"data\VOCtrainval_06-Nov-2007\VOCdevkit"

model = yolov1()
model.load_weights("yolo_temp.weights.h5") if False else None

ids = load_image_ids(voc_root)
image_id = ids[0]

image = load_image(voc_root, image_id)
input_tensor = np.expand_dims(image, axis = 0)

pred = model.predict(input_tensor)[0]
boxes = decode_predictions(pred, conf_threshold = 0.001)

output_img = draw_boxes((image*225).astype(np.uint8),boxes)

pred = model.predict(image[None, ...])[0]

print("RAW BOX (0,0):", pred[0,0,0:4])
print("RAW BOX (3,3):", pred[3,3,0:4])
print("RAW BOX (6,6):", pred[6,6,0:4])

cv2.imshow("Yolo v1 Predictions" , output_img)
cv2.waitKey(0)
cv2.destroyAllWindows
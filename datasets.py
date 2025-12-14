import os 
import cv2
import numpy as np 
import xml.etree.ElementTree as ET

img_size = 448
s = 7
b = 2

classes= ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"]

def load_image_ids (voc_root):
    path = os.path.join(voc_root , "VOC2007", "ImageSets" , "Main", "trainval.txt")
    with open(path) as f:
        return[line.strip() for line in f.readlines()]
    
def load_image(voc_root , image_id):
    path = os.path.join(voc_root , "VOC2007" , "JPEGImages" , image_id + ".jpg")
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img/255.0

def parse_annotation(voc_root, image_id):
    path = os.path.join(voc_root, "VOC2007", "Annotations" , image_id + ".xml")
    tree = ET.parse(path)
    root = tree.getroot()

    boxes, labels = [] , []

    for obj in root.findall("onject"):
        label = obj.find("name").text
        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels
"""
def encode_yolo(boxes,labels):
    target = np.zeros((s,s,b*5 + len(classes)))
    cell = img_size/s

    for box,label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        i = int(cx/cell)
        j = int(cy/cell)

        x = (cx - i*cell) / cell
        y = (cy - j*cell) / cell

        target[j , i, 0:5] = [x , y , w/img_size , h / img_size , 1]
        class_idx = classes.index(label)
        target[j , i, b*5 + class_idx] = 1

        return target 
 """   
def encode_yolo_target(boxes, labels, img_size=448, S=7, B=2, C=20):
    target = np.zeros((S, S, B*5 + C), dtype=np.float32)

    cell_size = img_size / S

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

        i = int(y_center / cell_size)
        j = int(x_center / cell_size)

        x_cell = (x_center - j * cell_size) / cell_size
        y_cell = (y_center - i * cell_size) / cell_size

        w_norm = w / img_size
        h_norm = h / img_size
        
        target[i, j, 0:5] = [x_cell, y_cell, w_norm, h_norm, 1.0]
        target[i, j, 5 + label] = 1.0
        break

    return target


import numpy as np 
import cv2

img_size = 448
s = 7
b = 2
c = 20

classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"]

def decode_predictions(pred, conf_threshold = 0.001):
    boxes = []
    cell_size = img_size/s

    for j in range(s):
        for i in range(s):
            cell = pred[j,i]

            confidence = cell[4]
            if confidence < conf_threshold:
                continue
            x,y,w,h = cell[0:4]
            class_probs = cell[10:]
            class_id = np.argmax(class_probs)
            score = confidence * class_probs[class_id]

            cx = (i+x)*cell_size
            cy = (j+y)*cell_size
            bw = w*img_size
            bh = h*img_size

            xmin = int(cx-bw/2)
            ymin = int(cy - bh /2)
            xmax = int(cx+bw / 2)
            ymax = int(cy + bh / 2)

            boxes.append([xmin, ymin, xmax, ymax, score, class_id])

        return boxes

def draw_boxes(image, boxes):
    img = image.copy()

    for box in boxes:
        xmin, xmax,ymin,ymax, score, class_id = box
        label = classes[class_id]

        cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (0,255,0) , 2)
        cv2.outText(img, f"{label} {score:.2f}",(xmin, max(ymin - 5, 0)), cv2.FRONT_HERSHEY_SIMPLEX,0.5,
            (255,255,255),1)
    return img


import cv2
import numpy as np

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2

image_path = "car.jpg"  # Replace with the path to your image

classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

img = cv2.imread(image_path)
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward()

classIds = []
confs = []
bbox = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > thres:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            classIds.append(classId)
            confs.append(float(confidence))
            bbox.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

for i in indices:
    i = i[0]
    box = bbox[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    cv2.putText(img, classNames[classIds[i]].upper(), (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

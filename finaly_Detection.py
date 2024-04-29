import cv2
import numpy as np

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2
url = "http://192.168.0.102:8080/video"
cap = cv2.VideoCapture(url)

classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('n').split('n')

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success, img = cap.read()
    if not success:
        break

    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward()

    classIds = []
    confs = []
    bbox = []

    for out in outs:
        for detection in out:
            if len(detection) < 6:
                continue  # Skip invalid detections
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] if classId < len(scores) else 0
            if np.any(confidence > thres):
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    w = int(detection[2] * img.shape[1])
                    h = int(detection[3] * img.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    classIds.append(classId)
                    confs.append(float(confidence))
                    bbox.append([x, y, w, h])
            else:
                print("invalid class" , classId)


    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classIds[i]].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import heapq
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#Store clicked Points on image
clicked_points = []

image = cv2.imread('car.jpg')
img = image * 0

height, width, _ = image.shape

# Find the coordinates of the white boxes

coordinates = []
def coordinates_inside_box(min_x, min_y, max_x, max_y):
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            coordinates.append((x, y))



def OnClickMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            cv2.circle(image , (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Bounding Box" , image)
            print(len(clicked_points))
            print(clicked_points)

# Get the indices of the unconnected output layers
output_layers_indices = net.getUnconnectedOutLayers()

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layer_names = [layer_names[index - 1] for index in output_layers_indices]

# Load classes
with open("yolov3.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Read the original image

# Convert image to blob
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set input blob for the network
net.setInput(blob)

# Run forward pass to get outputs
outs = net.forward(output_layer_names)

# Initialize variables to store coordinates of the entire detected object
min_x, min_y = float('inf'), float('inf')
max_x, max_y = 0, 0

# Iterate through detections to find min and max coordinates
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            # Get bounding box coordinates
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Update min and max coordinates
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

coordinates_inside_box(min_x , min_y , max_x,max_y)

# Draw single bounding box covering the entire detected object
cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
cv2.rectangle(img , (min_x , min_y), (max_x , max_y) , (255,255,255) , -1)
cv2.imshow("hello" , img)
cv2.imshow('Bounding Box', image)

white_coords = np.where(image == 255)
print(white_coords)

def is_inside_box(point):
    """
    for i in range(len(coordinates[0])):
        if point[1] in coordinates and point[0] in coordinates:
            return True
    return False"""
    if((point[0] >= min_x and point[0] <= max_x) and (point[1] >= min_y and point[1] <= max_y)):
        return True
    return False

# Function to compute the Euclidean distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def shortest_path(image, start, end):
    visited = set()
    heap = [(0, start, [])]
    mini = start
    pth = []
    vist = set()
    green_flag = 1
    while True:
        if mini == end:
            return pth
        x, y = mini
        min_dist = max(width, height)
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for neighbor in neighbors:
            if(not is_inside_box(neighbor)):
                nt1 = (neighbor[0] + 20, neighbor[1])
                nt2 = (neighbor[0], neighbor[1] + 20)
                nt3 = (neighbor[0] + 20, neighbor[1] + 20)
                nt4 = (neighbor[0], neighbor[1] - 20)
                nt5 = (neighbor[0] - 20, neighbor[1])
                if(is_inside_box(nt1) or is_inside_box(nt2) or is_inside_box(nt3) or is_inside_box(nt4) or is_inside_box(nt5)):
                    #cv2.circle(image , nt2, 2, (255, 0, 0), -1)
                    print(neighbor , nt2 , sep=" ")
                    continue
                if(distance(neighbor , end) < min_dist ):
                    min_dist = distance(neighbor, end)
                    mini = neighbor
        if mini in vist:
            continue
        vist.add(mini)
        pth = pth + [mini]

    return None

cv2.setMouseCallback("Bounding Box" , OnClickMouse)
if len(clicked_points) == 2:
    print("DONE")
    start_point = clicked_points[0]
    end_point = clicked_points[1]
    print(start_point)
    print(end_point)

cv2.waitKey(0)
cv2.destroyAllWindows()

path = shortest_path(image, clicked_points[0], clicked_points[1])
print(path)
# Draw lines connecting the points on the shortest path
for i in range(len(path) - 1):
    cv2.line(image, path[i], path[i + 1], (0, 0, 255), 1)

print("DDOOnne")
#print(coordinates)
cv2.imshow('Shortest Path', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import math

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = float('inf')  # Cost from start node
        self.h = float('inf')  # Heuristic (Manhattan distance to goal)
        self.f = float('inf')  # Total cost (g + h)


def manhattan_distance(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)

def astar(start, goal, grid):
    open_list = []
    closed_list = []

    start.g = 0
    start.h = manhattan_distance(start, goal)
    start.f = start.g + start.h

    open_list.append(start)

    while open_list:
        current = min(open_list, key=lambda node: node.f)

        if current == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]  # Reverse path to start -> goal order

        open_list.remove(current)
        closed_list.append(current)

        for neighbor in get_neighbors(current, grid):
            if neighbor in closed_list:
                continue
            if avoid_obstacle(neighbor):
                continue

            tentative_g = current.g + 1  # Assuming uniform cost for each step

            if neighbor not in open_list or tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = manhattan_distance(neighbor, goal)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_list:
                    open_list.append(neighbor)

    return None  # No path found

def get_neighbors(node, grid):
    neighbors = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        x, y = node.x + dx, node.y + dy
        if 0 <= x < len(grid[0]) and 0 <= y < len(grid) and grid[y][x] is not None:
            neighbors.append(grid[y][x])
    return neighbors

def avoid_obstacle(pt):
    if min_x == float('inf') or min_y == float('inf') or max_x == 0 or max_y == 0:
        return False  # If no object is detected, no obstacle

    min_center = node_centers[min(math.floor(min_y / 40), len(node_centers) - 1)][min(math.floor(min_x / 40), len(node_centers[0]) - 1)]
    max_center = node_centers[min(math.floor(max_y / 40), len(node_centers) - 1)][min(math.floor(max_x / 40), len(node_centers[0]) - 1)]
    
    point = [pt.x * grid_size + grid_size // 2, pt.y * grid_size + grid_size // 2]

    if ((point[0] >= min_center[0] and point[0] <= max_center[0]) and (point[1] >= min_center[1] and point[1] <= max_center[1])):
        return True
    else:
        return False

# Function to handle mouse clicks for specifying start and end points
def mouse_click(event, x, y, flags, param):
    global start_point, end_point
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = grid_centers[y // grid_size][x // grid_size]
        print("Start point set to:", start_point.x, start_point.y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        end_point = grid_centers[y // grid_size][x // grid_size]
        print("End point set to:", end_point.x, end_point.y)

# Load the video feed
video_capture = cv2.VideoCapture(0)

# Create a window and set mouse callback to handle click events
cv2.namedWindow("Grid with Circles and Path")
cv2.setMouseCallback("Grid with Circles and Path", mouse_click)

while True:
    # Capture a frame
    ret, frame = video_capture.read()
    
    # Use the frame as input
    img = frame.copy()
    path_image = frame.copy()

    # Get the indices of the unconnected output layers
    output_layers_indices = net.getUnconnectedOutLayers()

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[index - 1] for index in output_layers_indices]

    # Read the original image

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set input blob for the network
    net.setInput(blob)

    # Run forward pass to get outputs
    outs = net.forward(output_layer_names)

    # Initialize variables to store coordinates of the entire detected object
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    object_detected = False

    # Iterate through detections to find min and max coordinates
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                object_detected = True
                # Get bounding box coordinates
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                w = int(detection[2] * img.shape[1])
                h = int(detection[3] * img.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Update min and max coordinates
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

    # Draw single bounding box covering the entire detected object if detected
    if object_detected:
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.rectangle(path_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    else:
        cv2.putText(img, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Get the dimensions of the image
    h, w, _ = img.shape

    # Define the grid size (cell width and height)
    grid_size = 40

    # Initialize a list to store grid centers
    grid_centers = []
    node_centers = []

    # Draw grid lines and circles
    for y in range(grid_size // 2, h, grid_size):
        row_centers = []  # List to store centers for current row
        rc = []
        for x in range(grid_size // 2, w, grid_size):
            cv2.circle(img, (x, y), 2, (255, 0, 255), 2)
            row_centers.append(Node(x // grid_size, y // grid_size))  # Store center point as node
            rc.append((x, y))
        grid_centers.append(row_centers)  # Store row of center points
        node_centers.append(rc)

    # Draw grid lines
    for i in range(0, w + 1, grid_size):
        cv2.line(img, (i, 0), (i, h), (255, 0, 0), 2, -1)

    for i in range(0, h + 1, grid_size):
        cv2.line(img, (0, i), (w, i), (0, 255, 255), 2, -1)

    # Convert clicked points to grid coordinates
    # (Manually set start and end points by clicking on the grid)
    start_node = start_point if 'start_point' in globals() else grid_centers[0][0]
    end_node = end_point if 'end_point' in globals() else grid_centers[-1][-1]

    # Calculate the number of rows and columns in the grid
    num_rows = len(grid_centers)
    num_cols = len(grid_centers[0])

    # Create a grid of nodes
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    for y, row in enumerate(grid_centers):
        for x, node in enumerate(row):
            grid[y][x] = node

    # Find the shortest path using A*
    path = astar(start_node, end_node, grid)

    if path:
        # Draw path on the image
        for i in range(len(path)-1):
            # Calculate coordinates of the bounding rectangle
            x1 = min(path[i][0], path[i + 1][0]) * grid_size
            y1 = min(path[i][1], path[i + 1][1]) * grid_size
            x2 = max(path[i][0], path[i + 1][0]) * grid_size + grid_size
            y2 = max(path[i][1], path[i + 1][1]) * grid_size + grid_size
            # Draw rectangle between the nodes along the path
            cv2.rectangle(path_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.line(path_image, (path[i][0] * grid_size + grid_size // 2, path[i][1] * grid_size + grid_size // 2),(path[i+1][0] * grid_size + grid_size // 2, path[i+1][1] * grid_size + grid_size // 2), (255,255,255), 2, -1)

    # Display the image with grid, circles, detected object bounding box, and the shortest path
    cv2.imshow("Grid with Circles and Path", img)
    cv2.imshow("Path Image", path_image)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
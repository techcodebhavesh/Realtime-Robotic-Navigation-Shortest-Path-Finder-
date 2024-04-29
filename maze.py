import cv2
import numpy as np
import heapq

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f

def astar(maze, start, end):
    # Create start and end nodes
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize open and closed lists
    open_list = []
    closed_list = []

    # Add the start node
    heapq.heappush(open_list, (0, start_node))

    # Define possible movements (up, down, left, right)
    neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Loop until the end node is found
    while open_list:
        # Get the current node
        current_node = heapq.heappop(open_list)[1]

        # Add current node to closed list
        closed_list.append(current_node)

        # Check if end node is reached
        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        # Generate children nodes
        children = []
        for new_position in neighbors:
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Check if the node is within the maze bounds
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Check if the node is part of the border or already in closed list
            if maze[node_position[0]][node_position[1]] == 255 or any(node_position == child.position for child in closed_list):
                continue

            # Create a new node
            new_node = Node(current_node, node_position)

            # Calculate the g, h, and f values
            new_node.g = current_node.g + 1
            new_node.h = ((new_node.position[0] - end_node.position[0]) ** 2) + ((new_node.position[1] - end_node.position[1]) ** 2)
            new_node.f = new_node.g + new_node.h

            # Check if the node is already in the open list
            for i, (f, node) in enumerate(open_list):
                if new_node == node and new_node.g >= node.g:
                    break
            else:
                # Add the new node to the open list
                heapq.heappush(open_list, (new_node.f, new_node))

    # No path found
    return None

# Global variables to store the clicked points
clicked_points = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Maze Borders', image)
            print(len(clicked_points))

def main():
    global image

    # Load the image
    tp = cv2.imread('test.jpg')
    image = cv2.resize(tp, (int(tp.shape[1] / 2), int(tp.shape[0] / 2)))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Invert binary image
    binary = cv2.bitwise_not(binary)

    # Find contours to get the maze borders
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a 2D grid representing the maze
    maze_grid = np.zeros_like(gray)

    # Fill the detected borders on the grid
    cv2.drawContours(maze_grid, contours, -1, (255), thickness=cv2.FILLED)

    # Display the image with the detected borders
    cv2.imshow('Maze Borders', maze_grid)
    cv2.setMouseCallback('Maze Borders', mouse_callback)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate shortest path if two points are selected
    if len(clicked_points) == 2:
        print("DONE")
        start_point = clicked_points[0]
        end_point = clicked_points[1]
        print(start_point)
        print(end_point)

        # Find the shortest path avoiding the maze border
        path = astar(maze_grid, start_point, end_point)
        print(path)

        if path:
            # Draw the path segments on a copy of the original image
            image_with_path = image.copy()
            for i in range(len(path) - 1):
                pt1, pt2 = path[i], path[i+1]
                # Check if the distance between the two points is greater than 1
                if np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) > 1:
                    # If so, interpolate between the points to create intermediate points
                    num_intermediate_points = int(np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))
                    x_values = np.linspace(pt1[0], pt2[0], num_intermediate_points)
                    y_values = np.linspace(pt1[1], pt2[1], num_intermediate_points)
                    intermediate_points = [(int(x), int(y)) for x, y in zip(x_values, y_values)]
                    for i in range(len(intermediate_points) - 1):
                        cv2.line(image_with_path, intermediate_points[i], intermediate_points[i+1], (0, 255, 0), 2)
                else:
                    cv2.line(image_with_path, pt1, pt2, (0, 255, 0), 2)

            # Display the image with the path
            cv2.imshow('Shortest Path', image_with_path)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No path found.")

if __name__ == "__main__":
    main()

import matplotlib.pylab as plt
from skimage.morphology import skeletonize
import numpy as np
import cv2

img_name = 'maze.jpg'
rgb_img = plt.imread(img_name)

tp = cv2.imread('maze.jpg')
maze_img = cv2.resize(tp , ( int(tp.shape[1] / 2) , int(tp.shape[0] / 2)))

clickedPoints = []
def mouse_callback(event , x, y, flags, param):
    global clickedPoints
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clickedPoints) < 2:
            clickedPoints.append((x,y))
            cv2.circle(maze_img , (x,y) , 3, [0,255,0], -1)
            cv2.imshow('img', maze_img)
            print(clickedPoints)


cv2.imshow("img" , maze_img)
cv2.setMouseCallback("img" , mouse_callback)
cv2.waitKey(0)
print("done")
cv2.destroyWindow("img")

plt.figure(figsize=(14, 14))
x0, y0 = clickedPoints[0][0] * 2 , clickedPoints[0][1] * 2  # start x point
x1, y1 = clickedPoints[1][0] * 2 , clickedPoints[1][1] * 2 # start y point

plt.plot(x0, y0, 'gx', markersize=14)
plt.plot(x1, y1, 'rx', markersize=14)

if rgb_img.shape.__len__() > 2:
    thr_img = rgb_img[:, :, 0] > np.max(rgb_img[:, :, 0]) / 2
else:
    thr_img = rgb_img > np.max(rgb_img) / 2
skeleton = skeletonize(thr_img)
plt.figure(figsize=(14, 14))
# map of routes.
mapT = ~skeleton

plt.plot(x0, x0, 'gx', markersize=14)
plt.plot(x1, y1, 'rx', markersize=14)

_mapt = np.copy(mapT)

# searching for our end point and connect to the path.
boxr = 30

# Just a little safety check, if the points are too near the edge, it will error.
if y1 < boxr: y1 = boxr
if x1 < boxr: x1 = boxr

cpys, cpxs = np.where(_mapt[y1 - boxr:y1 + boxr, x1 - boxr:x1 + boxr] == 0)
# calibrate points to main scale.
cpys += y1 - boxr
cpxs += x1 - boxr
# find clooset point of possible path end points
idx = np.argmin(np.sqrt((cpys - y1) ** 2 + (cpxs - x1) ** 2))
y, x = cpys[idx], cpxs[idx]

pts_x = [x]
pts_y = [y]
pts_c = [0]

# mesh of displacements.
xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
ymesh = ymesh.reshape(-1)
xmesh = xmesh.reshape(-1)

dst = np.zeros((thr_img.shape))

# Breath first algorithm exploring a tree
while (True):
    # update distance.
    idc = np.argmin(pts_c)
    ct = pts_c.pop(idc)
    x = pts_x.pop(idc)
    y = pts_y.pop(idc)
    # Search 3x3 neighbourhood for possible
    ys, xs = np.where(_mapt[y - 1:y + 2, x - 1:x + 2] == 0)
    # Invalidate these point from future searchers.
    _mapt[ys + y - 1, xs + x - 1] = ct
    _mapt[y, x] = 9999999
    # set the distance in the distance image.
    dst[ys + y - 1, xs + x - 1] = ct + 1
    # extend our list.s
    pts_x.extend(xs + x - 1)
    pts_y.extend(ys + y - 1)
    pts_c.extend([ct + 1] * xs.shape[0])
    # If we run of points.
    if pts_x == []:
        break
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < boxr:
        edx = x
        edy = y
        break
plt.figure(figsize=(14, 14))


path_x = []
path_y = []

y = edy
x = edx
# Traces best path
while (True):
    nbh = dst[y - 1:y + 2, x - 1:x + 2]
    nbh[1, 1] = 9999999
    nbh[nbh == 0] = 9999999
    # If we reach a deadend
    if np.min(nbh) == 9999999:
        break
    idx = np.argmin(nbh)
    # find direction
    y += ymesh[idx]
    x += xmesh[idx]

    if np.sqrt((x - x1) ** 2 + (y - y1) ** 2) < boxr:
        print('Optimum route found.')
        break
    path_y.append(y)
    path_x.append(x)

plt.figure(figsize=(14, 14))
plt.imshow(rgb_img)
plt.plot(path_x, path_y, 'r-')  # Remove linewidth parameter

# Make a copy of the original image and draw the path on it
final_img = np.copy(rgb_img)
final_img[path_y, path_x] = [255, 0, 0]  # Red color for the path

# Draw the path with OpenCV to control line width
for i in range(len(path_x) - 1):
    cv2.line(final_img, (path_x[i], path_y[i]), (path_x[i+1], path_y[i+1]), (255, 0, 0), thickness=15)  # Set thickness here

# Convert the image to BGR format (required by OpenCV for displaying)
resized_img = cv2.resize(final_img, (int(final_img.shape[1] / 2), int(final_img.shape[0] / 2)))

# Convert the image to BGR format (required by OpenCV for displaying)
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)

cv2.imshow('Final Image with Path (Resized)', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

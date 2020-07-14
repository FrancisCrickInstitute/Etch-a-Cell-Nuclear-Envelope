import math
import cv2 as cv
import numpy as np

minVArea = 0.0001
maxVArea = 0.5

origImage = cv.imread("../../resources/Tiled3x3/C1166R0_140117_Run1_0036.tif", cv.IMREAD_GRAYSCALE)
totArea = origImage.size
image = cv.normalize(origImage, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)


def getClosestContour(x, y):
    contour = []
    matches = [contour for contour in contours if cv.pointPolygonTest(contour, (x, y), False) >= 0]
    mindist = -1
    for match in matches:
        m = cv.moments(match)
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist < mindist or mindist < 0:
            mindist = dist
            contour = match
    if mindist >= 0:
        return contour
    return []


def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        contour = getClosestContour(x, y)
        if len(contour):
            cv.drawContours(dispImage, [contour], -1, (0, 0, 255), 2)   # using list draws line


image = 1 - image
image = cv.GaussianBlur(image, (3, 3), 1)
image = np.uint8(image * 255)
thres, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

# Find contours
contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = [contour for contour in contours if minVArea < contour.size / totArea < maxVArea]

dispImage = cv.cvtColor(origImage.copy(), cv.COLOR_GRAY2BGR)

cv.namedWindow('image')
cv.setMouseCallback('image', mouse_click)

while True:
    cv.imshow('image', dispImage)
    k = cv.waitKey(20) & 0xFF
    if k == 27:
        cv.destroyAllWindows()
        break

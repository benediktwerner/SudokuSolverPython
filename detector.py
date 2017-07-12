import cv2
import numpy as np
from Board import Board

def rectify_contours(contour):
    contour = contour.reshape((4,2))
    result = np.zeros((4,2), np.float32)

    add = contour.sum(1)
    result[0] = contour[np.argmin(add)]
    result[2] = contour[np.argmax(add)]

    diff = np.diff(contour, axis=1)
    result[1] = contour[np.argmin(diff)]
    result[3] = contour[np.argmax(diff)]
    return result

def wk():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_number(image, name):
    _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    if not rects:
        return 0
    x,y,w,h = max(rects, key=lambda r: r[2]*r[3])
    length = round(h * 1.6)
    new_x = round(length/2 - w/2)
    new_y = round(length/2 - h/2)
    centered = np.zeros((length, length), np.uint8)
    centered[new_y:new_y+h, new_x:new_x+w] = image[y:y+h, x:x+w]

    #DEBUG
    cv2.imshow(name, centered)
    return 1
    #DEBUG END

    #centered = cv2.erode(centered, (11, 11))
    scaled = cv2.resize(centered, (28, 28), interpolation=cv2.INTER_AREA)
    _,scaled = cv2.threshold(scaled, 64, 255, cv2.THRESH_BINARY)
    #scaled = cv2.erode(scaled, (7,7))
    prediction = 5 # predict(scaled)
    return int(prediction[0])

def get_corner(image, xDir, yDir):
    for i in range(30):
        for j in range(i+1):
            x = (i - j) if xDir == 1 else (449 - i + j)
            y = j if yDir == 1 else (449 - j)
            if image[y, x] == 255:
                return (x, y)

def get_corners(image):
    corners = []
    corners.append(get_corner(image,  1,  1))
    corners.append(get_corner(image, -1,  1))
    corners.append(get_corner(image, -1, -1))
    corners.append(get_corner(image,  1, -1))
    return corners

def threshold(image):
    image = cv2.GaussianBlur(image, (11,11), 0)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

def detect_sudoku(image):
    sudoku = cv2.imread(image, 0)
    sudoku = threshold(sudoku)

    # Find largest blob
    _, contours, _ = cv2.findContours(sudoku.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = sudoku.shape
    max_area = (min(width, height)/2)**2 # assume the board covers at least a quater of the screen
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            arcLength = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*arcLength, True) # aproximate polygon
            if len(approx) == 4: # if rectangle
                max_area = area
                max_contour = approx

    # Rectify board
    max_contour = rectify_contours(max_contour)
    h =  np.array(((0,0), (449,0), (449,449), (0,449)), np.float32)
    ret = cv2.getPerspectiveTransform(max_contour, h)
    warp = cv2.warpPerspective(sudoku, ret, (450,450))

    cv2.imshow("warp", warp)

    # Remove border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _,border = cv2.threshold(warp, 64, 255, cv2.THRESH_BINARY)
    border = cv2.dilate(border, kernel, iterations=2)
    _,border = cv2.threshold(border, 64, 255, cv2.THRESH_BINARY)
    cv2.imshow("dilated", border)

    cv2.floodFill(border, np.zeros((452, 452), np.uint8), (0, 0), 64)
    for y, row in enumerate(border):
        for x, pixel in enumerate(row):
            if pixel != 64:
                border[y, x] = 255
            else:
                border[y, x] = 0

    cv2.imshow("border", border)
    without_border = cv2.bitwise_and(warp, border)
    cv2.imshow("without border", without_border)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    without_border = cv2.erode(without_border, kernel2, iterations=1)
    cv2.imshow("eroded", without_border)

    # Remove border and leftover noise
    dilate = cv2.dilate(without_border, kernel2, iterations=1)
    cv2.imshow("after dilate", dilate)
    opened = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, (3, 3))
    cv2.imshow("after open", opened)

    # Rewarp image without border
    corners = np.array(get_corners(border), np.float32)
    # for c in corners:
    #     cv2.rectangle(opened, (c[0]-1,c[1]-1), (c[0]+1,c[1]+1), 255)
    # cv2.imshow("corners", opened)
    h = np.array(((0,0), (449,0), (449,449), (0,449)), np.float32)
    ret = cv2.getPerspectiveTransform(corners, h)
    final = cv2.warpPerspective(opened, ret, (450,450))
    # _,final = cv2.threshold(final, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("final", final)

    sudoku_board = []
    for y in range(9):
        row = []
        for x in range(9):
            row.append(detect_number(final[y*50:(y+1)*50, x*50:(x+1)*50], "{} {}".format(x,y)))
        sudoku_board.append(row)
        if y > 2: return wk()

    return Board(9, sudoku_board)

print(detect_sudoku("sudoku.jpg"))

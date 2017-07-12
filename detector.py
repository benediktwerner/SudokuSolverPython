import cv2
import numpy as np
from Board import Board

def rectify_contour(contour):
    contour = contour.reshape((4,2))
    result = np.zeros((4,2), np.float32)

    add = contour.sum(1)
    result[0] = contour[np.argmin(add)]
    result[2] = contour[np.argmax(add)]

    diff = np.diff(contour, axis=1)
    result[1] = contour[np.argmin(diff)]
    result[3] = contour[np.argmax(diff)]
    return result

def find_corner(image, xDir, yDir):
    for i in range(30):
        for j in range(i+1):
            x = (i - j) if xDir == 1 else (449 - i + j)
            y = j if yDir == 1 else (449 - j)
            if image[y, x] == 255:
                return (x, y)

def find_corners(image):
    corners = []
    corners.append(find_corner(image,  1,  1))
    corners.append(find_corner(image, -1,  1))
    corners.append(find_corner(image, -1, -1))
    corners.append(find_corner(image,  1, -1))
    return corners

def threshold(image):
    image = cv2.GaussianBlur(image, (11,11), 0)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

def find_sudoku(image):
    '''Find largest blob'''
    _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # assume the board covers at least a quater of the screen
    max_area = (min(image.shape)/2)**2
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            arcLength = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*arcLength, True) # aproximate polygon
            if len(approx) == 4: # if rectangle
                max_area = area
                max_contour = approx
    return max_contour

def extract_sudoku(image, contour):
    contour = rectify_contour(contour)
    h = np.array(((0,0), (449,0), (449,449), (0,449)), np.float32)
    ret = cv2.getPerspectiveTransform(contour, h)
    return cv2.warpPerspective(image, ret, (450,450))

def find_border(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _,border = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
    border = cv2.dilate(border, kernel, iterations=2)
    _,border = cv2.threshold(border, 64, 255, cv2.THRESH_BINARY)

    cv2.floodFill(border, np.zeros((452, 452), np.uint8), (0, 0), 64)
    for y, row in enumerate(border):
        for x, pixel in enumerate(row):
            if pixel != 64:
                border[y, x] = 255
            else:
                border[y, x] = 0
    return border

def clean_image(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.erode(image, kernel, iterations=1)
    cv2.imshow("after erode", image)
    image = cv2.dilate(image, kernel, iterations=1)
    cv2.imshow("after dilate", image)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, (3, 3))
    cv2.imshow("after open", image)
    return image

def remove_border(image):
    contour = find_sudoku(image)
    sudoku = extract_sudoku(image, contour)

    cv2.imshow("extracted sudoku", sudoku)

    border = find_border(sudoku)
    cv2.imshow("border", border)

    sudoku = cv2.bitwise_and(sudoku, border)
    cv2.imshow("without border", sudoku)

    sudoku = clean_image(sudoku)

    # Rewarp image without border
    corners = np.array(find_corners(border), np.float32)
    # for c in corners:
    #     cv2.rectangle(sudoku, (c[0]-1,c[1]-1), (c[0]+1,c[1]+1), 255)
    # cv2.imshow("corners", sudoku)
    h = np.array(((0,0), (449,0), (449,449), (0,449)), np.float32)
    ret = cv2.getPerspectiveTransform(corners, h)
    return cv2.warpPerspective(sudoku, ret, (450,450))

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

def detect_numbers(image):
    board = []
    for y in range(9):
        row = []
        for x in range(9):
            row.append(detect_number(image[y*50:(y+1)*50, x*50:(x+1)*50], "{} {}".format(x,y)))
        board.append(row)
        if y > 2: return board

def detect_sudoku(image):
    sudoku = cv2.imread(image, 0)
    cv2.imread("original", sudoku)
    sudoku = threshold(sudoku)
    sudoku = remove_border(sudoku)

    board = detect_numbers(sudoku)
    return "DEBUG MODE"
    return Board(9, board)

print(detect_sudoku("sudoku.jpg"))
cv2.waitKey(0)
cv2.destroyAllWindows()

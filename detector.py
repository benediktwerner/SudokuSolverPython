import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from Board import Board

def rectify(contour):
    contour = contour.reshape((4,2))
    result = np.zeros((4,2), np.float32)

    add = contour.sum(1)
    result[0] = contour[np.argmin(add)]
    result[2] = contour[np.argmax(add)]

    diff = np.diff(contour, axis=1)
    result[1] = contour[np.argmin(diff)]
    result[3] = contour[np.argmax(diff)]
    return result

def detect_number(image, name):
    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    if not rects:
        return 0
    x,y,w,h = max(rects, key=lambda r: r[2]*r[3])
    length = round(h * 1.6)
    new_x = round(length/2 - w/2)
    new_y = round(length/2 - h/2)
    centered = np.zeros((length, length), np.uint8)
    centered[new_y:new_y+h, new_x:new_x+w] = image[y:y+h, x:x+w]
    #centered = cv2.erode(centered, (11, 11))
    scaled = cv2.resize(centered, (28, 28), interpolation=cv2.INTER_AREA)
    _,scaled = cv2.threshold(scaled, 64, 255, cv2.THRESH_BINARY)
    #scaled = cv2.erode(scaled, (7,7))
    hog_fd = hog(scaled, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    prediction = clf.predict(np.array([hog_fd], np.float64))
    return int(prediction[0])

def detect_sudoku(image):
    sudoku = cv2.imread("sudoku.jpg")
    sudoku_gray = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
    height,width = sudoku_gray.shape

    # Blurr and threshold
    sudoku_gray = cv2.GaussianBlur(sudoku_gray, (11,11), 0)
    sudoku_thresh = cv2.adaptiveThreshold(sudoku_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, contours, _ = cv2.findContours(sudoku_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest blob
    max_area = (min(width, height)/2)**2 # assume the board covers at least a quater of the screen
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            arcLength = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*arcLength, True)
            if len(approx) == 4:
                max_area = area
                max_contour = approx

    # Rectify board
    max_contour = rectify(max_contour)
    h =  np.array(((0,0), (449,0), (449,449), (0,449)), np.float32)
    ret = cv2.getPerspectiveTransform(max_contour, h)
    warp = cv2.warpPerspective(sudoku_thresh, ret, (450,450))

    cv2.imshow("Sudoku", warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

    # Remove border
    cv2.imshow("before", warp)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _,border = cv2.threshold(warp, 64, 255, cv2.THRESH_BINARY)
    border = cv2.dilate(warp, kernel, iterations=2)
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
    # cv2.imshow("before", warp)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # warp = cv2.dilate(warp, kernel, iterations=1)
    # cv2.imshow("after 1", warp)
    # cv2.floodFill(warp, np.zeros((452, 452), np.uint8), (0,0), 0)
    # warp = cv2.erode(warp, (5, 5), iterations=4)
    # opened = cv2.morphologyEx(warp, cv2.MORPH_OPEN, (3, 3))

    clf = joblib.load("digits_cls.pkl")
    sudoku_board = []
    for y in range(9):
        row = []
        for x in range(9):
            row.append(detect_number(without_border[y*50:(y+1)*50, x*50:(x+1)*50], "{} {}".format(x,y)))
        sudoku_board.append(row)

    print(Board(9, sudoku_board))

    # cv2.imshow("Sudoku Warp", warp)
    # cv2.imshow("Sudoku Opened", opened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_sudoku("sudoku.jpg")

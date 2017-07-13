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
    #cv2.imshow("after erode", image)
    image = cv2.dilate(image, kernel, iterations=1)
    #cv2.imshow("after dilate", image)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, (3, 3))
    #cv2.imshow("after open", image)
    return image

def remove_border(image):
    contour = find_sudoku(image)
    sudoku = extract_sudoku(image, contour)
    #cv2.imshow("extracted sudoku", sudoku)
    border = find_border(sudoku)
    #cv2.imshow("border", border)
    sudoku = cv2.bitwise_and(sudoku, border)
    #cv2.imshow("without border", sudoku)
    sudoku = clean_image(sudoku)

    # Rewarp image without border
    corners = np.array(find_corners(border), np.float32)
    # for c in corners:
    #     cv2.rectangle(sudoku, (c[0]-1,c[1]-1), (c[0]+1,c[1]+1), 255)
    # cv2.imshow("corners", sudoku)
    h = np.array(((0,0), (449,0), (449,449), (0,449)), np.float32)
    ret = cv2.getPerspectiveTransform(corners, h)
    return cv2.warpPerspective(sudoku, ret, (450,450))

def detect_sudoku(image):
    sudoku = cv2.imread(image, 0)
    #cv2.imshow("original", sudoku)
    sudoku = threshold(sudoku)
    sudoku = remove_border(sudoku)

    board = NumberDetector().detect_numbers(sudoku)
    return Board(board)

class NumberDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = False
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.maxConvexity = 1

        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def center_digit(self, image, rect):
        length = rect.height #round(rect.height * 1.6)
        new_x = round(length/2 - rect.width/2)
        new_y = round(length/2 - rect.height/2)
        centered = np.zeros((length, length), np.uint8)
        p1 = rect.p1()
        p2 = rect.p2()
        centered[new_y:new_y+rect.height, new_x:new_x+rect.width] = image[p1[1]:p2[1], p1[0]:p2[0]]
        return centered

    def detect_digit(self, image):
        _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [Rect(*cv2.boundingRect(contour)) for contour in contours]
        rect = max(rects, key=lambda r: r.area(), default=None)
        if not rect or rect.area() < 100:
            return 0
        image = self.center_digit(image, rect)

        blobs = self.blob_detector.detect(image)
        #print("Blobs: ", *map(lambda x:"point: {} size: {}".format(x.pt, x.size), blobs), sep="  -  ", end="\n")

        # image_with_keys = cv2.drawKeypoints(image, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.rectangle(image_with_keys, rect.p1(), rect.p2(), (0,255,0))
        # cv2.imshow("digit keypoints", image_with_keys)
        cv2.imshow("digit", image)
        cv2.waitKey(0)

        blobs_count = len(blobs)
        if blobs_count == 2:
            return 8
        if blobs_count == 1:
            return self.detect_1_blob(blobs[0], image.shape)
        if blobs_count == 0:
            return self.detect_no_blobs(image, rect)
        return 0

    def detect_1_blob(self, blob, shape):
        blob_y = blob.pt[1] / shape[0]
        if blob_y <= 0.45:
            return 9
        if blob_y > 0.55:
            return 6
        return 4

    def detect_no_blobs(self, image, rect):
        if rect.ratio() <= 0.63:
            return 1

        center = self.find_center(image)
        height = self.find_biggest_height(image)

        if height > 0.5:
            return 3
        if center[1] <= 0.42:
            return 7
        if (center[0] < 0.49 and center[1] < 0.49 and height > 0.41) or height > 0.44:
            return 5
        return 2

    def find_center(self, image):
        total_x = 0
        total_y = 0
        n = 0
        for y, row in enumerate(image):
            for x, pixel in enumerate(row):
                if pixel > 128:
                    total_x += x
                    total_y += y
                    n += 1
        h, w = image.shape
        return (total_x / n / w, total_y / n / h)

    def find_biggest_height(self, image):
        h, w = image.shape
        max_height = 0
        for x in range(w):
            curr_height = 0
            for y in range(h):
                if image[y, x] > 128:
                    curr_height += 1
                elif curr_height > 0:
                    max_height = max(max_height, curr_height)
                    curr_height = 0
        return max_height / h

    def detect_numbers(self, image):
        board = []
        for y in range(9):
            row = []
            for x in range(9):
                row.append(self.detect_digit(image[y*50:(y+1)*50, x*50:(x+1)*50]))
            board.append(row)
        return board

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def p1(self):
        return (self.x, self.y)

    def p2(self):
        return (self.x + self.width, self.y + self.height)

    def area(self):
        return self.width * self.height

    def ratio(self):
        return self.width / self.height

print(detect_sudoku("sudoku2.png"))
cv2.waitKey(0)
cv2.destroyAllWindows()

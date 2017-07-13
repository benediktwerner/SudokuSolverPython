import cv2
import numpy as np
from math import copysign
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
    cv2.imshow("border thresh 1", border)
    border = cv2.dilate(border, kernel, iterations=1)
    cv2.imshow("border dilate", border)
    _,border = cv2.threshold(border, 64, 255, cv2.THRESH_BINARY)
    cv2.imshow("border thresh 2", border)

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
    cv2.imshow("extracted sudoku", sudoku)
    border = find_border(sudoku)
    cv2.imshow("border", border)
    sudoku = cv2.bitwise_and(sudoku, border)
    sudoku = clean_image(sudoku)

    # Rewarp image without border
    corners = find_corners(border)
    corners_array = np.array(corners, np.float32)
    # corners_image = sudoku.copy()
    # for c in corners:
    #     cv2.rectangle(corners_image, (c[0]-1,c[1]-1), (c[0]+1,c[1]+1), 255)
    # cv2.imshow("corners", corners_image)
    h = np.array(((0,0), (449,0), (449,449), (0,449)), np.float32)
    ret = cv2.getPerspectiveTransform(corners_array, h)
    return cv2.warpPerspective(sudoku, ret, (450,450))

def detect_sudoku(image):
    sudoku = cv2.imread(image, 0)
    if sudoku is None:
        return None
    sudoku = threshold(sudoku)
    cv2.imshow("threshold", sudoku)
    sudoku = remove_border(sudoku)
    cv2.imshow("borderless", sudoku)

    board = NumberDetector().detect_numbers(sudoku)
    cv2.destroyAllWindows()
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

    def find_rect(self, image):
        _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [Rect(*cv2.boundingRect(contour)) for contour in contours]
        return max(rects, key=lambda r: r.area(), default=None)

    def extract_digit(self, image):
        _,image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
        h,w = image.shape
        x = round(w/2)
        for y in range(h//2, round(h*0.85)):
            if image[y, x] > 230:
                cv2.floodFill(image, np.zeros((w+2, h+2), np.uint8), (x, y), 64)
                break
            return None
        for y, row in enumerate(image):
            for x, pixel in enumerate(row):
                if pixel == 64:
                    image[y, x] = 255
                elif pixel != 0:
                    image[y, x] = 0
        return image

    def detect_digit(self, image, repeat=True):
        rect = self.find_rect(image)
        if not rect or rect.area() < 100 or rect.width > rect.height or rect.x > 30 or rect.y > 30 or rect.x + rect.width < 25 or rect.y + rect.height < 25:
            return 0

        image = self.center_digit(image, rect)
        if image[0, 0] > 200 or rect.area() > 1600:
            if not repeat:
                return 0
            image = self.extract_digit(image)
            if image is None:
                return 0
            return self.detect_digit(image, repeat=False)

        blobs = self.blob_detector.detect(image)
        #print("Blobs: ", *map(lambda x:"point: {} size: {}".format(x.pt, x.size), blobs), sep="  -  ", end="\n")

        # image_with_keys = cv2.drawKeypoints(image, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.rectangle(image_with_keys, rect.p1(), rect.p2(), (0,255,0))
        # cv2.imshow("digit keypoints", image_with_keys)
        # cv2.imshow("digit", image)
        # cv2.waitKey(0)

        blobs_count = len(blobs)
        if blobs_count == 2:
            return 8
        if blobs_count == 1:
            return self.detect_1_blob(image, blobs[0], image.shape)
        if blobs_count == 0:
            return self.detect_no_blobs(image, rect)
        return 0

    def detect_1_blob(self, image, blob, shape):
        blob_y = blob.pt[1] / shape[0]
        if blob_y <= 0.44:
            return 9
        if blob_y > 0.6:
            return 6
        if blob_y < 0.52:
            return 4
        right_outline = self.get_right_outline(image)
        if right_outline == [1]:
            return 4
        return 6

    def detect_no_blobs(self, image, rect):
        if rect.ratio() <= 0.63:
            return 1

        left_outline = self.get_left_outline(image)
        right_outline = self.get_right_outline(image)

        if len(left_outline) > 5 or left_outline == [-1, 1, -1, 1, -1]:
            return 3
        if right_outline == [-1, 1, -1] or right_outline == [1, -1] or right_outline == [1]:
            return self.detect_3_7(image)
        if left_outline == [-1, 1, -1, 1] and (right_outline == [1, -1, 1] or right_outline == [-1, 1, -1, 1]):
            return 2
        if left_outline == [1, -1, 1, -1] or left_outline == [-1, 1, -1]:
            return self.detect_2_5(image)
        print("Weird outlines:")
        print(left_outline)
        print(right_outline)
        print(rect)
        cv2.imshow("wierd outlines", image)
        cv2.waitKey(0)
        return 0

        # Outlines:
        # 2:
        #   -1, 1, -1 // 1, -1, 1, -1
        #   -1, 1, -1 // 1, -1, 1
        #   -1, 1, -1, 1 // 1, -1, 1
        #
        # 3:
        #   -1, 1, -1, 1, -1 //
        #                       1
        #                       1, -1
        #                       -1, 1
        #                       -1, 1, -1
        # 5:
        #   (1), -1, 1, -1 // ((-1)), 1, -1, 1, (-1)
        # 7:
        #   -1, 1, -1 // (-1), 1, -1

    def detect_3_7(self, image):
        h, w = image.shape
        m = round(w / 2)
        switch = 0
        curr = 0
        for y in range(h):
            if image[y, m] > 230:
                if curr == 0:
                    switch += 1
                    curr = 1
            elif curr == 1:
                switch += 1
                curr = 0
        if switch < 5:
            return 7
        return 3

    def detect_2_5(self, image):
        h,w = image.shape
        for x, pixel in enumerate(reversed(image[round(h/4)])):
            if pixel > 230:
                if x < w / 2:
                    return 2
                else:
                    return 5
        return -2

    def get_left_outline(self, image):
        outline = []
        curr = 0
        m = round(image.shape[1] / 2)
        for row in image:
            for x, pixel in enumerate(row):
                if pixel > 230:
                    pos = int(copysign(1, x - m))
                    if pos != curr:
                        outline.append(pos)
                        curr = pos
                    break
        return outline

    def get_right_outline(self, image):
        outline = []
        curr = 0
        m = round(image.shape[1] / 2)
        for row in image:
            for x, pixel in enumerate(reversed(row)):
                if pixel > 230:
                    pos = int(copysign(1, m - x))
                    if pos != curr:
                        outline.append(pos)
                        curr = pos
                    break
        return outline

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

    def __str__(self):
        return "({}, {}, {}, {})".format(self.x, self.y, self.width, self.height)

if __name__ == "__main__":
    print(detect_sudoku("sudoku.jpg"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

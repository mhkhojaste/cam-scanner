import cv2
import numpy as np
import imutils


def delete_extra_rects(rects):
    min_arr = np.amin(rects[0], axis=0)
    max_arr = np.amax(rects[0], axis=0)
    h = 100

    print(min_arr)
    print(max_arr)
    i = 0
    while len(rects[0]) > 4:
        print("len : ", len(rects[0]))
        print("i : ", i)
        print(rects[0][i])
        if i == 4:
            rects[0] = np.delete(rects[0], [i], axis=0)
            continue
        if not (abs(rects[0][i][0] - min_arr[0]) <= h and abs(rects[0][i][1] - min_arr[1]) <= h):
            print("1")
            if not (abs(rects[0][i][0] - min_arr[0]) <= h and abs(rects[0][i][1] - max_arr[1]) <= h):
                print("2")
                if not (abs(rects[0][i][0] - max_arr[0]) <= h and abs(rects[0][i][1] - max_arr[1]) <= h):
                    print("3")
                    if not (abs(rects[0][i][0] - max_arr[0]) <= h and abs(rects[0][i][1] - min_arr[1]) <= h):
                        print("4")
                        rects[0] = np.delete(rects[0], [i], axis=0)
                        continue

        i += 1
    return rects


if __name__ == '__main__':
    image_path = "test1.jpg"
    orig = cv2.imread(image_path)
    MORPH = 9
    CANNY = 84
    HOUGH = 25
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(img, (3, 3), 0, img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    dilated = cv2.dilate(img, kernel)
    edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, 3.14 / 180, HOUGH)
    for line in lines[0]:
        cv2.line(edges, (line[0], line[1]), (line[2], line[3]),
                 (255, 0, 0), 2, 8)

    # finding contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter(lambda cont: cv2.arcLength(cont, False) > 1000, contours)
    # simplify contours down to polygons
    rects = []
    for cont in contours:
        rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
        rects.append(rect)
    while len(rects) > 1:
        del rects[1]

    cv2.drawContours(orig, rects, -1, (0, 255, 0), 1)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', orig)
    cv2.waitKey(0)

    # fix the corner points. ex : 0 L 50 : it change the point 0 to Left for 50
    while True:
        st = input()
        if st == "end":
            break
        index = int(st[0])
        if st[2] == "L":
            rects[0][index][0] -= int(st[4:])
        if st[2] == "R":
            rects[0][index][0] += int(st[4:])
        if st[2] == "U":
            rects[0][index][1] -= int(st[4:])
        if st[2] == "D":
            rects[0][index][1] += int(st[4:])

        orig = cv2.imread(image_path)
        cv2.drawContours(orig, rects, -1, (0, 255, 0), 1)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', orig)
        cv2.waitKey(0)

    # delete extra rects
    rects = delete_extra_rects(rects)
    # transformation
    rows, cols, ch = orig.shape
    widthA = np.sqrt(((rects[0][0][0] - rects[0][3][0]) ** 2) + ((rects[0][0][1] - rects[0][3][1]) ** 2))
    widthB = np.sqrt(((rects[0][1][0] - rects[0][2][0]) ** 2) + ((rects[0][1][1] - rects[0][2][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((rects[0][0][0] - rects[0][1][0]) ** 2) + ((rects[0][0][1] - rects[0][1][1]) ** 2))
    heightB = np.sqrt(((rects[0][3][0] - rects[0][2][0]) ** 2) + ((rects[0][3][1] - rects[0][2][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(np.float32(rects), dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    warped = cv2.flip(warped, 1)
    warped = imutils.rotate_bound(warped, 180)
    width1 = int(warped.shape[1] * 4)
    height1 = int(warped.shape[0] * 1.2)
    dim = (width1, height1)
    # resize image
    warped = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("warped", warped)
    cv2.waitKey(0)

    # thresholding
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mean_c = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
    cv2.imshow('meanc', mean_c)
    cv2.waitKey(0)
    # sharpening
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(mean_c, -1, kernel_sharpening)
    cv2.imshow('Image Sharpening', sharpened)
    cv2.imwrite('final.jpg', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

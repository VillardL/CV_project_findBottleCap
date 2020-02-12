import math

import cv2
import numpy as np
from PIL.Image import fromarray
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
# import vgg

import classify


def division(img):
    # 分别储存颜色和正反信息的两个Set
    capcolor = img[20:4000, 20:3000]  # 拍照的边造成不可控融合，所以同意把边切了
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(gradient, kernel)
    for i in range(3):
        dilated = cv2.dilate(dilated, kernel)
    (_, thresh) = cv2.threshold(dilated, 60, 255, cv2.THRESH_BINARY)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    kernel3 = np.ones((10, 10), np.uint8)
    erosion = cv2.erode(closed, kernel3, iterations=1)
    blur = cv2.medianBlur(erosion, 31)
    dilate = cv2.dilate(blur, kernel3)
    close1 = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel2)
    capall = close1[20:4000, 20:3000]
    gq = fromarray(np.uint8(capall))
    gq.save("capA.jpg")
    img1 = cv2.imread('capA.jpg')
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, bipix = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img, contours, hierarchy = cv2.findContours(bipix, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, capcolor


def judge(contours, capcolor):
    colorSet = []
    pnSet = []
    capSet = []
    all = len(contours)
    for i in range(all):
        cnt = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
        if (radius > 240 and radius < 300):
            x1 = x - radius
            x2 = x + radius
            y1 = y - radius
            y2 = y + radius
            capThis = capcolor[y1:y2, x1:x2]
            name = "littlecap" + str(i) + ".jpg"
            img_RGB = cv2.cvtColor(capThis, cv2.COLOR_BGR2RGB)
            # 分类
            clf = classify.classify(img_RGB)  # 返回0/1

            gqi = fromarray(np.uint8(img_RGB))
            gqi.save(name)
            cv2.rectangle(capcolor, (x1, y1), (x2, y2), (238, 104, 123), thickness=7)
            # vgg.pred(name, "./")
            rect = cv2.minAreaRect(cnt)  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            p1 = box[0]
            p2 = box[1]
            p3 = box[2]
            p4 = p2 - p1
            p5 = p3 - p2
            p6 = math.hypot(p4[0], p4[1])
            p7 = math.hypot(p5[0], p5[1])
            colorSet.append(color(img_RGB))
            if abs(p6 - p7) > 100:
                capSet.append(i)
                pnSet.append("立")
            else:
                capSet.append(i)
                if clf == 0:
                    pnSet.append("正")
                else:
                    pnSet.append("反")
    # 因为目前测试图片问题，如果没有10个完全被识别展示没什么用，可以看命令行里的输出
    # print(colorSet)
    # print(capSet)
    # print(pnSet)
    rec = fromarray(np.uint8(cv2.cvtColor(capcolor, cv2.COLOR_BGR2RGB)))
    rec.save("rec.jpg")

    return capSet, pnSet, colorSet


def judge_ResNet(contours, capcolor):
    colorSet = []
    pnSet = []
    capSet = []
    all = len(contours)
    for i in range(all):
        cnt = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
        if (radius > 240 and radius < 300):
            x1 = x - radius
            x2 = x + radius
            y1 = y - radius
            y2 = y + radius
            capThis = capcolor[y1:y2, x1:x2]
            name = "littlecap" + str(i) + ".jpg"
            img_RGB = cv2.cvtColor(capThis, cv2.COLOR_BGR2RGB)
            # 分类
            clf = classify.classify(img_RGB)  # 返回0/1

            gqi = fromarray(np.uint8(img_RGB))
            gqi.save(name)
            rect = cv2.minAreaRect(cnt)  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            p1 = box[0]
            p2 = box[1]
            p3 = box[2]
            p4 = p2 - p1
            p5 = p3 - p2
            p6 = math.hypot(p4[0], p4[1])
            p7 = math.hypot(p5[0], p5[1])
            if abs(p6 - p7) > 100:
                capSet.append(i)
                pnSet.append("立")
            else:
                capSet.append(i)
                if clf == 0:
                    pnSet.append("正")
                else:
                    pnSet.append("反")
    # 因为目前测试图片问题，如果没有10个完全被识别展示没什么用，可以看命令行里的输出
    # print(colorSet)
    # print(capSet)
    # print(pnSet)
    return (capSet, pnSet)


def judge_VGG(contours, capcolor):
    colorSet = []
    pnSet = []
    capSet = []
    all = len(contours)
    for i in range(all):
        cnt = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
        if (radius > 240 and radius < 300):
            x1 = x - radius
            x2 = x + radius
            y1 = y - radius
            y2 = y + radius
            capThis = capcolor[y1:y2, x1:x2]
            name = "littlecap" + str(i) + ".jpg"
            img_RGB = cv2.cvtColor(capThis, cv2.COLOR_BGR2RGB)
            # 分类
            clf = classify.classify(img_RGB)  # 返回0/1

            gqi = fromarray(np.uint8(img_RGB))
            gqi.save(name)
            rect = cv2.minAreaRect(cnt)  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            p1 = box[0]
            p2 = box[1]
            p3 = box[2]
            p4 = p2 - p1
            p5 = p3 - p2
            p6 = math.hypot(p4[0], p4[1])
            p7 = math.hypot(p5[0], p5[1])
            if abs(p6 - p7) > 100:
                capSet.append(i)
                pnSet.append("立")
            else:
                capSet.append(i)
                if clf == 0:
                    pnSet.append("正")
                else:
                    pnSet.append("反")
    # 因为目前测试图片问题，如果没有10个完全被识别展示没什么用，可以看命令行里的输出
    # print(colorSet)
    # print(capSet)
    # print(pnSet)
    return (capSet, pnSet)


def point_color(p):
    if p[0] > 180 and p[1] > 180 and p[2] < 170:
        return 3
    elif p[0] > 180 and p[1] < 170 and p[2] > 180:
        return 4
    elif p[0] < 170 and p[1] > 180 and p[2] > 180:
        return 5
    elif p[0] > 200 and p[1] > 200 and p[2] > 200:
        return 6
    elif p[0] < 50 and p[1] < 50 and p[2] < 50:
        return 7
    elif abs(p[0] - p[1]) < 10 and abs(p[0] - p[2]) < 10:
        return 8
    elif max(p[0], p[1], p[2]) == p[0]:
        return 0
    elif max(p[0], p[1], p[2]) == p[1]:
        return 1
    elif max(p[0], p[1], p[2]) == p[2]:
        return 2


# 0:R, 1:G, 2:B, 3:Yellow, 4:Purple, 5:Cyan, 6:White, 7:Black, 8:Gray


def prior_color(list):
    r = g = b = y = p = c = w = bl = gr = 0
    for i in range(len(list)):
        # print(list[i])
        if list[i] == 0:
            r += 1
        elif list[i] == 1:
            g += 1
        elif list[i] == 2:
            b += 1
        elif list[i] == 3:
            y += 1
        elif list[i] == 4:
            p += 1
        elif list[i] == 5:
            c += 1
        elif list[i] == 6:
            w += 1
        elif list[i] == 7:
            bl += 1
        elif list[i] == 8:
            gr += 1
    sum_pc = (r + g + b + y + p + c + w + bl + gr) * 0.6
    if max(r, g, b, y, p, c, w, bl, gr) == r and r > sum_pc:
        return 0
    elif max(r, g, b, y, p, c, w, bl, gr) == g and g > sum_pc:
        return 1
    elif max(r, g, b, y, p, c, w, bl, gr) == b and b > sum_pc:
        return 2
    elif max(r, g, b, y, p, c, w, bl, gr) == y and y > sum_pc:
        return 3
    elif max(r, g, b, y, p, c, w, bl, gr) == p and p > sum_pc:
        return 4
    elif max(r, g, b, y, p, c, w, bl, gr) == c and c > sum_pc:
        return 5
    elif max(r, g, b, y, p, c, w, bl, gr) == w and w > sum_pc:
        return 6
    elif max(r, g, b, y, p, c, w, bl, gr) == bl and bl > sum_pc:
        return 7
    elif max(r, g, b, y, p, c, w, bl, gr) == gr and gr > sum_pc:
        return 8
    else:
        return -1


def color(img_RGB):
    print(img_RGB)
    print(len(img_RGB))
    l = len(img_RGB)
    o_x = o_y = int(l / 2)
    cycle_colors = []
    ok = True
    count = 0
    for r in range(70, int(l / 2)):
        count += 1
        point_colors = []
        for x in range(r):
            y = int(math.sqrt(r * r - x * x))
            point_colors.append(point_color(img_RGB[o_x + x][o_y + y]))
            point_colors.append(point_color(img_RGB[o_x + x][o_y - y]))
            point_colors.append(point_color(img_RGB[o_x - x][o_y + y]))
            point_colors.append(point_color(img_RGB[o_x - x][o_y - y]))
        cycle_colors.append(prior_color(point_colors))
        ok = True
        if count > 5:
            for i in range(1, 4):
                if cycle_colors[len(cycle_colors) - 1 - i] != cycle_colors[len(cycle_colors) - 1] or cycle_colors[
                    len(cycle_colors) - 1 - i] == -1:
                    ok = False
            if ok:
                return cycle_colors[len(cycle_colors) - 1]
    return 0

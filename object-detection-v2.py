import cv2
import time
import pyautogui
import math

import numpy as np
import imutils as im

from mss import mss
from Recorder import Recorder
from scipy.spatial import distance as dist


class GameObject:
    def __init__(self, center, radius, contour, isMainChar):
        self.center = center
        self.radius = radius
        self.contour = contour
        self.isMainChar = isMainChar


def nothing(x):
    pass


def create_trackbars(threshold_value=20, blur_size=15):
    cv2.namedWindow('VALUES')

    # 20 initial value, optimal for now for black background, we can play with this to get better result
    # use trackbar to find better threshold value
    cv2.createTrackbar('threshold_value', 'VALUES', threshold_value, 255, nothing)
    cv2.createTrackbar('blur_size', 'VALUES', blur_size, 20, nothing)

    cv2.moveWindow('VALUES', 0, 0)

    val = np.ones((100, 350, 3), np.uint8)
    cv2.imshow('VALUES', val * 255)


def process(image, virus, base_color, save=False):
    # threshold_value = cv2.getTrackbarPos('threshold_value', 'VALUES')
    # blur_size = cv2.getTrackbarPos('blur_size', 'VALUES')

    # resize the image so processing will be faster
    resized_image = im.resize(image, width=image.shape[1])
    original_image = resized_image.copy()
    ratio = image.shape[1] / resized_image.shape[1]  # it is bigger than one

    # working to get the threshold image
    # if we do not resize our image the small dots will be appear on the processing image
    # we need to use better blur
    # blurred = cv2.GaussianBlur(resized_image, (2 * blur_size + 1, 2 * blur_size + 1), 0)
    # blurred = cv2.GaussianBlur(resized_image, (15, 15), 0) # if we are using resized image
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

    resized_image = cv2.bitwise_and(resized_image,resized_image,mask=thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if im.is_cv2() else cnts[1]

    height, width = resized_image.shape[:2]
    center_x, center_y = width // 2, height // 2

    min_dist = (np.inf, None)

    scene_objects = []

    # do whatever you want with contours
    for i, c in enumerate(cnts):
        M = cv2.moments(c)
        isMainChar = False
        if M['m00'] != 0:
            # if we want to show the original image
            # cX = int((M['m10'] / M['m00']) * ratio)
            # cY = int((M['m01'] / M['m00']) * ratio)

            # c = c.astype('float')
            # c *= ratio
            # c = c.astype('int')

            # cv2.drawContours(image, [c], -1, (0, 255, 0), -1)

            # if we want to show resized image
            cX = int((M['m10'] / M['m00']))
            cY = int((M['m01'] / M['m00']))

            #(x, y), radius = cv2.minEnclosingCircle(c)
            #cv2.circle(resized_image,(int(x),int(y)),int(radius),(255,255,0),2)
            area =  cv2.contourArea(c)
            radius = math.sqrt(area/math.pi)
            #epsilon = 0.1 * cv2.arcLength(c, True)
            #approx = cv2.approxPolyDP(c, epsilon, True)


            if np.array_equal(resized_image[cY,cX], base_color) and radius > 13:
                isMainChar = True

            d = dist.euclidean((center_x, center_y), (cX, cY))
            if d < min_dist[0] and radius > 13:
                min_dist = (d, len(scene_objects))

            go = GameObject(center=(cX, cY), radius=radius, contour=c, isMainChar=isMainChar)
            scene_objects.append(go)

            # making all of the green for now
            # recoloring according to their size will be implemented later
            # cv2.drawContours(resized_image, [c], -1, (0, 255, 0), -1)
            # cv2.putText(resized_image, str(int(radius)), (cX - 2, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

    if min_dist[1] is None:
        print('nothing found')
    else:
        self_object = scene_objects[min_dist[1]]
        for i, obj in enumerate(scene_objects):
            if i == min_dist[1] or obj.isMainChar:
                # self character
                cv2.drawContours(resized_image, [obj.contour], -1, (255, 255, 255), -1)
                #cv2.circle(resized_image, (int(obj.center[0]),int(obj.center[1])),int(obj.radius), (255,255,255), -1)

                # Template matching can be used
                # However, skin colors that are too close to black are problem
                #template_temp = cv2.resize(template,(int(2*obj.radius),int(2*obj.radius)))
                #if template_temp.shape[0] < gray.shape[0] and template_temp.shape[1] < gray.shape[1]:
                #    res = cv2.matchTemplate(gray, template_temp, cv2.TM_CCOEFF_NORMED)
                     # For matching more than 1 object
                #    w, h = template_temp.shape[::-1]
                #    threshold = 0.8
                #    loc = np.where(res >= threshold)
                #    for pt in zip(*loc[::-1]):
                #        cv2.rectangle(resized_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

                    # For matching 1 object
                    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    #top_left = max_loc
                    #bottom_right = (top_left[0] + 2*obj.radius, top_left[1] + 2*obj.radius)
                    #cv2.rectangle(resized_image,(int(top_left[0]),int(top_left[1])),(int(bottom_right[0]),int(bottom_right[1])),255,2)
                #else:
                #    template_temp = template.copy()
                #cv2.putText(resized_image, str(int(obj.radius)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

            else:
                if obj.radius * 118 / 100 < self_object.radius:
                    if np.array_equal(resized_image[obj.center[1], obj.center[0]], [51, 255, 0, 255]):
                        # might be virus
                        cv2.drawContours(resized_image, [obj.contour], -1, (0, 0, 255), -1)
                    else:
                        # can eat
                        cv2.drawContours(resized_image, [obj.contour], -1, (0, 255, 0), -1)
                        # cv2.putText(resized_image, str(int(obj.radius)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

                else:
                    # can not eat
                    cv2.drawContours(resized_image, [obj.contour], -1, (0, 0, 255), -1)
                    # cv2.putText(resized_image, str(int(obj.radius)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))


    #cv2.imshow('ORIGINAL', original_image)
    cv2.namedWindow('THRESH')
    cv2.moveWindow("THRESH", 0, window_height // 2)
    cv2.imshow('THRESH', thresh)
    cv2.namedWindow("FINAL")
    cv2.moveWindow("FINAL",window_width // 2,0)
    cv2.imshow('FINAL', resized_image)


    #if save:
        #recorder.Record([original_image, gray, thresh, resized_image])


if __name__ == '__main__':
    # creating environment variables
    window_width, window_height = pyautogui.size()

    sct = mss()
    roi = {'top': 97, 'left': 2, 'width': window_width // 2 - 2, 'height': window_height // 2}
    #roi = {'top': 0, 'left': 0, 'width': window_width, 'height': window_height}

    # create_trackbars()

    #global recorder
    #recorder = Recorder()

    #template = cv2.imread("skin.png",0)
    paused = False
    first_time = True
    base_color = []
    # wait 5 seconds to hide the terminal and start the game

    print("Open agar.io in 5 seconds")
    for i in range(5,0,-1):
        print(i)
        time.sleep(1)

    # main loop
    while True:
        if ~paused:
            t = time.time()
            image = np.array(sct.grab(monitor=roi), dtype='uint8')

            # For template matching use below function
            #process(image, template, ~paused)
            if first_time:
                base_color = image[image.shape[0]//2,image.shape[1]//2]
                first_time = False
            process(image, base_color, ~paused)
            #print(time.time() - t)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

        if key == ord('c'):
            paused = ~paused

        # wanted to move them eveytime so they wont interfere with region of interest
        # cv2.moveWindow('VALUES', 0, 0)
        # cv2.moveWindow('ORIGINAL', 0, 0)
        # cv2.moveWindow('THRESH', 0, window_height // 3)
        # cv2.moveWindow('FINAL', 0, 2 * window_height // 3)

    cv2.destroyAllWindows()
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
    def __init__(self, center, radius, area, contour):
        self.center = center
        self.radius = radius
        self.area = area
        self.contour = contour
        #self.isMainObj = isMainObj


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


def process(image, base_color, save=False):
    # threshold_value = cv2.getTrackbarPos('threshold_value', 'VALUES')
    # blur_size = cv2.getTrackbarPos('blur_size', 'VALUES')
    num_of_objects = 0
    # resize the image so processing will be faster
    resized_image = im.resize(image, width=image.shape[1])
    original_image = resized_image.copy()
    ratio = image.shape[1] / resized_image.shape[1]  # it is bigger than one

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

    resized_image = cv2.bitwise_and(resized_image,resized_image,mask=thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if im.is_cv2() else cnts[1]

    height, width = resized_image.shape[:2]
    center_x, center_y = width // 2, height // 2

    min_dist = (np.inf, None)
    #min_dist_scene = (np.inf, None)

    scene_objects = []
    main_objects = []

    # do whatever you want with contours
    for i, c in enumerate(cnts):

        M = cv2.moments(c)
        #isMainObj = False
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

            area =  cv2.contourArea(c)
            radius = math.sqrt(area/math.pi)

            #epsilon = 0.1 * cv2.arcLength(c, True)
            #approx = cv2.approxPolyDP(c, epsilon, True)

            d = dist.euclidean((center_x, center_y), (cX, cY))
            if d < min_dist[0] and radius > 13:
                min_dist = (d, len(main_objects))
                print("main_objects",len(main_objects))
                #min_dist_scene = (d, len(scene_objects))

            go = GameObject(center=(cX, cY), radius=radius, area=area, contour=c)

            if np.array_equal(resized_image[cY,cX], base_color) and radius > 13:
                #isMainObj = True
                main_objects.append(go)
            else:
                scene_objects.append(go)

    if min_dist[1] is None:
        print('nothing found')
    else:
        num_of_objects = len(main_objects)
        # TODO: main_object sometines is empty, when two divided object combining
        #if num_of_objects == 0:
        #    main_objects.append(scene_objects[min_dist_scene[1]])
        #    del scene_objects[min_dist_scene[1]]
        #    num_of_objects = len(main_objects)

        main_obj = main_objects[min_dist[1]]  # main object nearest to center, probably biggest
        print(num_of_objects)
        # check for solidity, object may be divided but there is a collision between them
        object_area = cv2.contourArea(main_obj.contour)
        hull = cv2.convexHull(main_obj.contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(object_area) / hull_area

        if solidity < 0.9:
            num_of_objects = len(main_objects) + 1

        # if num_of_object is 1 and solidity below 0.90 then there is a collision
        cv2.putText(resized_image, str(round(solidity, ndigits=2)), (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        cv2.putText(resized_image, str(num_of_objects), (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

        for i, obj in enumerate(main_objects):
            # main character
            cv2.drawContours(resized_image, [obj.contour], -1, (255, 255, 255), -1)
            #cv2.putText(resized_image, str(int(obj.radius)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        for i, obj in enumerate(scene_objects):
            if obj.area * 139 / 100 < main_obj.area:
                if np.array_equal(resized_image[obj.center[1], obj.center[0]], [51, 255, 0, 255]) and obj.radius > 13:
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
    #cv2.namedWindow('THRESH')
    #cv2.moveWindow("THRESH", 0, window_height // 2)
    #cv2.imshow('THRESH', thresh)
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
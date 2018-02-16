import cv2
import math

import numpy as np
import imutils as im

from scipy.spatial import distance as dist

# could not make the other one work, so i used this one instead
# it was always failing and giving errors (main character not found and index is out of range at the line main_obj = main_objects[min_dist[1]])

class GameObject:
    def __init__(self, center, radius, area, contour):
        self.center = center
        self.radius = radius
        self.area = area
        self.contour = contour
        self.is_main = False
        self.piece = 1
        self.average_area = area
        #self.isMainObj = isMainObj

def process(image, base_color, config):
    # resize the image so processing will be faster
    resized_image = im.resize(image, width=image.shape[1] // 2)
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

    scene_objects = []

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

            d = dist.euclidean((center_x, center_y), (cX, cY))
            if d < min_dist[0]:
                min_dist = (d, len(scene_objects))

            go = GameObject(center=(cX, cY), radius=radius, area=area, contour=c)
            scene_objects.append(go)
    
    if min_dist[1] is None:
        print('nothing found')
    else:
        if base_color == []:
            x, y, w, h = cv2.boundingRect(scene_objects[min_dist[1]].contour)
            base_color = cv2.mean(resized_image[y : y + h, x : x + w])

        main_obj = None
        max_area = 0
        for (i, obj) in enumerate(scene_objects):
            # we need to look at the mean value of the image
            # when the pieces close to each other cannot find the main object most probably sees black
            # if main object is close to another object mean value wont work, it will say both are enemy
            x, y, w, h = cv2.boundingRect(obj.contour)
            difference_array = np.subtract(cv2.mean(resized_image[y : y + h, x : x + w]), base_color[0])
            #difference_array = np.subtract(resized_image[obj.center[1], obj.center[0]], base_color)
            result = np.less_equal(np.absolute(difference_array), [15, 15, 15, 5]) # change this values
            #print(difference_array)
            if np.all(result[:3]) and obj.radius > 10:
                obj.is_main = True

                if max_area < obj.average_area:
                    max_area = obj.average_area
                    main_obj = obj

        if main_obj is None:
            main_obj = scene_objects[min_dist[1]]
            main_obj.is_main = True
            #for (i, obj) in enumerate(scene_objects):
            #    cv2.drawContours(resized_image, [obj.contour], -1, (0, 0, 255), -1)
            #    #cv2.putText(resized_image, str(int(obj.radius)) + '/' + str(int(obj.piece)) + '/' + str(int(obj.average_area)) + '/' + str(int(obj.area)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255))

        for (i, obj) in enumerate(scene_objects):
            hull = cv2.convexHull(obj.contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(obj.area) / hull_area

            if solidity < .9:
                obj.piece += 1
                obj.average_area = obj.area / obj.piece

            if obj.is_main:
                cv2.drawContours(resized_image, [obj.contour], -1, (255, 255, 255), -1)
            else:
                # this should be other way around
                if obj.average_area * 139 / 100 < main_obj.average_area:
                    if np.array_equal(resized_image[obj.center[1], obj.center[0]], [51, 255, 0, 255]) and obj.radius > 10:
                        # might be virus
                        cv2.drawContours(resized_image, [obj.contour], -1, (0, 0, 255), -1)
                    else:
                        # can eat
                        cv2.drawContours(resized_image, [obj.contour], -1, (0, 255, 0), -1)
                else:
                    # can not eat
                    cv2.drawContours(resized_image, [obj.contour], -1, (0, 0, 255), -1)

            #cv2.putText(resized_image, str(int(obj.radius)) + '/' + str(int(obj.piece)) + '/' + str(int(obj.average_area)) + '/' + str(int(obj.area)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255))

    #cv2.drawContours(resized_image, [main_obj.contour], -1, (255, 255, 255), -1)
    #cv2.putText(resized_image, str(int(main_obj.radius)) + '/' + str(int(main_obj.piece)) + '/' + str(int(main_obj.average_area)) + '/' + str(int(main_obj.area)), (main_obj.center[0] - 2, main_obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255))
    
    #print('processing the scene is done')
    #cv2.imshow('ORIGINAL', original_image)
    #cv2.imshow('THRESH', thresh)
    #cv2.imshow('FINAL', resized_image)

    # consider returning grayscale image
    return cv2.cvtColor(im.resize(resized_image, width=config.sample_width, height=config.sample_height), cv2.COLOR_BGRA2GRAY), main_obj.area, base_color
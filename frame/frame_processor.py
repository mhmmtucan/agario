import math

import cv2
import imutils as im
import numpy as np
from scipy import stats
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
        self.nearest_enemy = GameObject
        #self.isMainObj = isMainObj

def process(image, base_color, config):
    # threshold_value = cv2.getTrackbarPos('threshold_value', 'VALUES')
    # blur_size = cv2.getTrackbarPos('blur_size', 'VALUES')

    num_of_objects = 0
    # resize the image so processing will be faster
    resized_image = im.resize(image, width=image.shape[1])

    #resized_image = cv2.resize(image, (0,0), fx=0.75, fy=0.75)
    #original_image = resized_image.copy()
    #ratio = image.shape[1] / resized_image.shape[1]  # it is bigger than one
    screen_center = (resized_image.shape[1] // 2 , resized_image.shape[0] // 2)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    resized_image = cv2.bitwise_and(resized_image,resized_image,mask=thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if im.is_cv2() else cnts[1]

    height, width = resized_image.shape[:2]
    center_x, center_y = width // 2, height // 2

    min_dist = (np.inf, None)
    min_dist_scene = (np.inf, None)
    nearest_enemy = (750, None)

    scene_objects = []
    main_objects = []

    # do whatever you want with contours
    for i, c in enumerate(cnts):
        colors = np.array((255,255,255,255),np.uint8)
        M = cv2.moments(c)
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

            go = GameObject(center=(cX, cY), radius=radius, area=area, contour=c)

            d = dist.euclidean((center_x, center_y), (cX, cY))

            if width/5 < cX and cX < width* 4/5 and height/5 < cY and cY < height * 4/5 :
                colors = np.array((resized_image[cY - 2, cX + 2], resized_image[cY - 2, cX - 2],
                                   resized_image[cY + 2, cX - 2], resized_image[cY + 2, cX + 2],
                                   resized_image[cY, cX + 2], resized_image[cY, cX - 2],
                                   resized_image[cY + 2, cX], resized_image[cY - 2, cX],
                                   resized_image[cY, cX]), np.uint8)
                colors = stats.mode(colors)[0][0]

            if len([el for el in [abs(base_color[i]) - abs(num) for i, num in enumerate(colors[:3])] if el < 5]) == 3 and radius > 13:
                # main object
                if d < min_dist[0] and radius > 13:
                    min_dist = (d, len(main_objects))
                main_objects.append(go)
                #cv2.putText(resized_image, str(base_color[0]) + " " + str(base_color[1]) + " " + str(base_color[2]),
                #            (5, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            else:
                # scene objects
                if d < min_dist_scene[0] and radius > 13:
                    min_dist_scene = (d, len(scene_objects))
                scene_objects.append(go)

    if min_dist_scene[1] is None and min_dist[1] is None:
        print('nothing found')
    else:
        num_of_objects = len(main_objects)
        if num_of_objects == 0:
            # main object collide with different color object
            main_objects.append(scene_objects[min_dist_scene[1]])
            del scene_objects[min_dist_scene[1]]
            num_of_objects = len(main_objects)

        main_obj = main_objects[num_of_objects-1]  # main object nearest to center, probably biggest

        # check for solidity, object may be divided but there is a collision between them
        object_area = cv2.contourArea(main_obj.contour)
        hull = cv2.convexHull(main_obj.contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(object_area) / hull_area

        if solidity < 0.9:
            num_of_objects = len(main_objects) + 1
            main_obj.area /= 2
            main_obj.radius /= 2
            '''
            for m_obj in main_objects:
                if not cv2.isContourConvex(m_obj.contour):
                    hull = cv2.convexHull(m_obj.contour, returnPoints=False)
                    defects = cv2.convexityDefects(m_obj.contour, hull)
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(m_obj.contour[s][0])
                        end = tuple(m_obj.contour[e][0])
                        far = tuple(m_obj.contour[f][0])
                        cv2.line(resized_image, start, end, [0, 255, 0], 2)
                        cv2.circle(resized_image, far, 5, [0, 0, 255], -1)
            '''
        # if num_of_object is 1 and solidity below 0.90 then there is a collision
        #cv2.putText(resized_image, "Solidity: "+str(round(solidity, ndigits=2)), (5, 20),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        #cv2.putText(resized_image, "Number of objects: "+str(num_of_objects), (5, 40),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

        for i, obj in enumerate(main_objects):
            # main character
            cv2.drawContours(resized_image, [obj.contour], 0, (255, 255, 255), -1)
            #cv2.putText(resized_image, str(int(obj.radius)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
            #cv2.putText(resized_image, "Object " + str(i+1) +  " radius: " + str(obj.radius // 1), (5, i*20+80),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

        for i, obj in enumerate(scene_objects):
            if obj.radius < 13 or (obj.area * 130 / 100 < main_obj.area and width/5 < obj.center[0] and obj.center[0] < width* 4/5 and
                    height/5 < obj.center[1] and obj.center[1] < height * 4/5):

                # can eat
                cv2.drawContours(resized_image, [obj.contour], 0, (0, 255, 0), -1)
                # cv2.putText(resized_image, str(int(obj.radius)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
            else:
                # can not eat
                d = dist.euclidean(main_obj.center, obj.center)
                if d < nearest_enemy[0]:
                    #main_obj.nearest_enemy = obj
                    nearest_enemy = (d,obj)
                cv2.drawContours(resized_image, [obj.contour], 0, (0, 0, 255), -1)

                # TODO: solve here
                #a = np.abs(np.array(resized_image[obj.center[1], obj.center[0]]) - np.array([30, 222, 30])) < [150, 150, 150]
                
                #print(np.all(a))
                #print(obj.radius)
                #if np.all(a) and obj.radius > 13:
                    # might be virus
                    #cv2.drawContours(resized_image, [obj.contour], 0, (0, 255, 255), -1)
                # cv2.putText(resized_image, str(int(obj.radius)), (obj.center[0] - 2, obj.center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

        #if nearest_enemy[0] != 750:
        #    cv2.line(resized_image, screen_center, nearest_enemy[1].center, (255, 255, 0), 5)

    return resized_image, cv2.cvtColor(cv2.resize(resized_image, (config.sample_width, config.sample_height)), cv2.COLOR_BGRA2GRAY), nearest_enemy[0]
import cv2
import time

import numpy as np

class Recorder:
    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = None

        self.h = None
        self.w = None
    
    def Record(self, images, recordResized=True):
        if self.writer is None:
            (self.h, self.w) = images[0].shape[:2]

            if recordResized:
                self.writer = cv2.VideoWriter("./videos/" + time.asctime(time.localtime(time.time())) + '.avi',
                                              self.fourcc, 10, (self.w , self.h), True)
            else:
                self.writer = cv2.VideoWriter("./videos/" + time.asctime(time.localtime(time.time())) + '.avi',
                                              self.fourcc, 10, (self.w * 2, self.h * 2), True)

        if recordResized:
            output = np.zeros((self.h, self.w, 3), dtype="uint8")
            if len(images[3].shape) == 2:
                images[3] = cv2.cvtColor(images[3], cv2.COLOR_GRAY2BGR)
            elif len(images[3].shape) == 3:
                if images[3].shape[2] == 4:
                    images[3] = cv2.cvtColor(images[3], cv2.COLOR_BGRA2BGR)
            output[0: self.h, 0: self.w] = images[3]

        else:
            output = np.zeros((self.h * 2, self.w * 2, 3), dtype="uint8")
            if len(images[0].shape) == 2:
                images[0] = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
            elif len(images[0].shape) == 3:
                if images[0].shape[2] == 4:
                    images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGRA2BGR)
            output[0: self.h, 0: self.w] = images[0]

            if len(images[1].shape) == 2:
                images[1] = cv2.cvtColor(images[1], cv2.COLOR_GRAY2BGR)
            elif len(images[1].shape) == 3:
                if images[1].shape[2] == 4:
                    images[1] = cv2.cvtColor(images[1], cv2.COLOR_BGRA2BGR)
            output[0: self.h, self.w: self.w * 2] = images[1]

            if len(images[2].shape) == 2:
                images[2] = cv2.cvtColor(images[2], cv2.COLOR_GRAY2BGR)
            elif len(images[2].shape) == 3:
                if images[2].shape[2] == 4:
                    images[2] = cv2.cvtColor(images[2], cv2.COLOR_BGRA2BGR)
            output[self.h: self.h * 2, 0: self.w] = images[2]

            if len(images[3].shape) == 2:
                images[3] = cv2.cvtColor(images[3], cv2.COLOR_GRAY2BGR)
            elif len(images[3].shape) == 3:
                if images[3].shape[2] == 4:
                    images[3] = cv2.cvtColor(images[3], cv2.COLOR_BGRA2BGR)
            output[self.h: self.h * 2, self.w: self.w * 2] = images[3]

        self.writer.write(output)
import cv2
import numpy as np

class Recorder:
    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.writer = None

        self.h = None
        self.w = None
    
    def Record(self, images):
        if self.writer is None:
            (self.h, self.w) = images[0].shape[:2]
            self.writer = cv2.VideoWriter('output.avi', self.fourcc, 10, (self.w * 2, self.h * 2), True)
        
        output = np.zeros((self.h * 2, self.w * 2, 3), dtype="uint8")
        if len(images[0].shape) ==  2:
            images[0] = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
        elif len(images[0].shape) == 3:
            if images[0].shape[2] == 4:
                images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGRA2BGR)
        output[0 : self.h, 0 : self.w] = images[0]

        if len(images[1].shape) ==  2:
            images[1] = cv2.cvtColor(images[1], cv2.COLOR_GRAY2BGR)
        elif len(images[1].shape) == 3:
            if images[1].shape[2] == 4:
                images[1] = cv2.cvtColor(images[1], cv2.COLOR_BGRA2BGR)
        output[0 : self.h, self.w : self.w * 2] = images[1]

        if len(images[2].shape) ==  2:
            images[2] = cv2.cvtColor(images[2], cv2.COLOR_GRAY2BGR)
        elif len(images[2].shape) == 3:
            if images[2].shape[2] == 4:
                images[2] = cv2.cvtColor(images[2], cv2.COLOR_BGRA2BGR)
        output[self.h : self.h * 2, 0: self.w] = images[2]

        if len(images[3].shape) ==  2:
            images[3] = cv2.cvtColor(images[3], cv2.COLOR_GRAY2BGR)
        elif len(images[3].shape) == 3:
            if images[3].shape[2] == 4:
                images[3] = cv2.cvtColor(images[3], cv2.COLOR_BGRA2BGR)
        output[self.h : self.h * 2, self.w : self.w * 2] = images[3]

        self.writer.write(output)
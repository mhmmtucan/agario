import os
import time

import numpy as np
import win32api as wapi

# could not find anything else
# for now i have used win32api for myself
# did not delete the old scritps
class InputCheck:
    def __init__(self, keys):
        self.keys = keys

        self.key_list = ['\b']
        for char in self.keys:
            self.key_list.append(char)

    def get_keys(self):
        keys = []
        for key in self.key_list:
            if wapi.GetAsyncKeyState(ord(key)):
                keys.append(key)
        return keys

class ExperienceBuffer:
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    # adding sample(s) to the experience buffer
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0 : (len(experience)  + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    # returning a sample from experience buffer
    def sample(self, size):
        # reshape it to [size, 6] to get size number of experiences with shape of [1, 6] -> past_areas, area, future_areas, mouse, space, frame
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 6])

    def length(self):
        return len(self.buffer)

    def save(self, filename):
        np.save(filename, self.buffer)
import platform
import random
import subprocess

import numpy as np
import pyautogui
from pynput.keyboard import Key
from pynput.keyboard import Listener
from scipy.spatial import distance as dist

platform_name = platform.system()

if platform_name == 'Windows':
    import win32api as wapi
elif platform_name == 'Linux':
    print('linux platform')
elif platform_name == 'Darwin':
    print('mac platform')
else:
    print('unrecognized platform')

def say(text):
    if platform_name == 'Darwin':
        subprocess.call('say ' + text, shell=True)
    else: pass

def get_unix_keys(queue):
    if platform_name != 'Windows':

        def on_release(key):
            unix_keys = []

            if key == Key.space:
                unix_keys.append(' ')
                queue.put(unix_keys)
            else:
                try:
                    unix_keys.append(key.char)
                    queue.put(unix_keys)
                    if key.char == 'q':
                        return False
                except: pass

        listener = Listener(on_release=on_release)
        listener.daemon = True
        listener.start()
        listener.join()

class Config:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.roi = {'top': 0, 'left': 0, 'width': self.screen_width, 'height': self.screen_height}
        self.center = (self.screen_width // 2, self.screen_height // 2)

        self.sample_width = 320
        self.sample_height = 180
        self.sample_depth = 1

        self.raw_width = 1280
        self.raw_height = 720
        self.raw_center = (self.raw_width // 2, self.raw_height // 2)
        self.raw_roi = {'top': 0, 'left': 0, 'width': self.raw_width, 'height': self.raw_height}
        self.raw_screen_ratio = self.raw_height / self.screen_height


        self.actions = 9 # dismissing space hits for now
        self.keys = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890'


class InputCheck:
    def __init__(self, config):
        self.keys = config.keys
        self.raw_width = config.raw_width
        self.raw_height = config.raw_height
        self.height_pieces = 3
        self.width_pieces = 3
        self.raw_screen_ratio = self.raw_height / config.screen_height

        self.key_list = ['\b']
        for char in self.keys:
            self.key_list.append(char)

    def get_keys(self):
        keys = []
        for key in self.key_list:
            if  platform_name == 'Windows':
                if wapi.GetAsyncKeyState(ord(key)):
                    keys.append(key)
        return keys

    def get_mouse_vector(self,mouse):
        vect = np.zeros((self.height_pieces,self.width_pieces),dtype=int)
        vect[mouse[1] // (self.raw_height // self.height_pieces)][mouse[0] // (self.raw_width // self.width_pieces)] = 1
        return vect.flatten().tolist()

    def get_mouse_position(self,vector):
        vect = np.array(vector).reshape(self.height_pieces,self.width_pieces)
        (y,x) = np.unravel_index(np.argmax(vect, axis=None), vect.shape)
        y_pos = int((y + 0.5) * (self.raw_height // self.height_pieces) * (1/self.raw_screen_ratio))
        x_pos = int((x + 0.5) * (self.raw_width // self.width_pieces) * (1/self.raw_screen_ratio))

        return tuple([x_pos,y_pos])


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


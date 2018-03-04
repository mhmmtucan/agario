import time
import random
import platform
import pyautogui
import numpy as np
import subprocess

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

        self.sample_width = 150
        self.sample_height = 90
        self.sample_depth = 1

        self.raw_width = 1280
        self.raw_height = 720
        self.raw_center = (self.raw_width // 2, self.raw_height // 2)
        self.raw_roi = {'top': 0, 'left': 0, 'width': self.raw_width, 'height': self.raw_height}
        self.raw_screen_ratio = self.raw_height / self.screen_height


        self.actions = 9 # dismissing space hits for now
        self.keys = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890'

# pygame and getkey modules did not work as i wanted. need to find another module for that
# could not find anything else
# for now i have used win32api for myself
# did not delete the old scritps
class InputCheck:
    def __init__(self, config):
        self.keys = config.keys

        self.key_list = ['\b']
        for char in self.keys:
            self.key_list.append(char)

        self.vector_to_positions = {
            (1, 0, 0, 0, 0, 0, 0, 0, 0): (config.center[0] - config.screen_width // 10, config.center[1] + config.screen_height // 10), # top left
            (0, 1, 0, 0, 0, 0, 0, 0, 0): (config.center[0], config.center[1] + config.screen_height // 10),                             # top
            (0, 0, 1, 0, 0, 0, 0, 0, 0): (config.center[0] + config.screen_width // 10, config.center[1] + config.screen_height // 10), # top right
            (0, 0, 0, 1, 0, 0, 0, 0, 0): (config.center[0] - config.screen_width // 10, config.center[1]),                              # mid left
            (0, 0, 0, 0, 1, 0, 0, 0, 0): (config.center[0], config.center[1]),                                                          # mid
            (0, 0, 0, 0, 0, 1, 0, 0, 0): (config.center[0] + config.screen_width // 10, config.center[1]),                              # mid right
            (0, 0, 0, 0, 0, 0, 1, 0, 0): (config.center[0] - config.screen_width // 10, config.center[1] - config.screen_height // 10), # bot left
            (0, 0, 0, 0, 0, 0, 0, 1, 0): (config.center[0], config.center[1] - config.screen_height // 10),                             # bot
            (0, 0, 0, 0, 0, 0, 0, 0, 1): (config.center[0] + config.screen_width // 10, config.center[1] - config.screen_height // 10)  # bot right
        }

        self.position_to_vector = {
            (config.center[0] - config.screen_width // 10, config.center[1] + config.screen_height // 10): (1, 0, 0, 0, 0, 0, 0, 0, 0), # top left
            (config.center[0], config.center[1] + config.screen_height // 10): (0, 1, 0, 0, 0, 0, 0, 0, 0),                             # top
            (config.center[0] + config.screen_width // 10, config.center[1] + config.screen_height // 10): (0, 0, 1, 0, 0, 0, 0, 0, 0), # top right
            (config.center[0] - config.screen_width // 10, config.center[1]): (0, 0, 0, 1, 0, 0, 0, 0, 0),                              # mid left
            (config.center[0], config.center[1]): (0, 0, 0, 0, 1, 0, 0, 0, 0),                                                          # mid
            (config.center[0] + config.screen_width // 10, config.center[1]): (0, 0, 0, 0, 0, 1, 0, 0, 0),                              # mid right
            (config.center[0] - config.screen_width // 10, config.center[1] - config.screen_height // 10): (0, 0, 0, 0, 0, 0, 1, 0, 0), # bot left
            (config.center[0], config.center[1] - config.screen_height // 10): (0, 0, 0, 0, 0, 0, 0, 1, 0),                             # bot
            (config.center[0] + config.screen_width // 10, config.center[1] - config.screen_height // 10): (0, 0, 0, 0, 0, 0, 0, 0, 1)  # bot right        
        }

        self.vector_to_positions_raw = {
            (1, 0, 0, 0, 0, 0, 0, 0, 0): (config.raw_center[0] - config.raw_width // 10, config.raw_center[1] + config.raw_height // 10),# top left
            (0, 1, 0, 0, 0, 0, 0, 0, 0): (config.raw_center[0], config.raw_center[1] + config.raw_height // 10),  # top
            (0, 0, 1, 0, 0, 0, 0, 0, 0): (config.raw_center[0] + config.raw_width // 10, config.raw_center[1] + config.raw_height // 10),# top right
            (0, 0, 0, 1, 0, 0, 0, 0, 0): (config.raw_center[0] - config.raw_width // 10, config.raw_center[1]),# mid left
            (0, 0, 0, 0, 1, 0, 0, 0, 0): (config.raw_center[0], config.raw_center[1]),  # mid
            (0, 0, 0, 0, 0, 1, 0, 0, 0): (config.raw_center[0] + config.raw_width // 10, config.raw_center[1]),# mid right
            (0, 0, 0, 0, 0, 0, 1, 0, 0): (config.raw_center[0] - config.raw_width // 10, config.raw_center[1] - config.raw_height // 10),# bot left
            (0, 0, 0, 0, 0, 0, 0, 1, 0): (config.raw_center[0], config.raw_center[1] - config.raw_height // 10),  # bot
            (0, 0, 0, 0, 0, 0, 0, 0, 1): (config.raw_center[0] + config.raw_width // 10, config.raw_center[1] - config.raw_height // 10)# bot right
        }

        self.position_to_vector_raw = {
            (config.raw_center[0] - config.raw_width // 10, config.raw_center[1] + config.raw_height // 10): (1, 0, 0, 0, 0, 0, 0, 0, 0),  # top left
            (config.raw_center[0], config.raw_center[1] + config.raw_height // 10): (0, 1, 0, 0, 0, 0, 0, 0, 0),  # top
            (config.raw_center[0] + config.raw_width // 10, config.raw_center[1] + config.raw_height // 10): (0, 0, 1, 0, 0, 0, 0, 0, 0),  # top right
            (config.raw_center[0] - config.raw_width // 10, config.raw_center[1]): (0, 0, 0, 1, 0, 0, 0, 0, 0),# mid left
            (config.raw_center[0], config.raw_center[1]): (0, 0, 0, 0, 1, 0, 0, 0, 0),  # mid
            (config.raw_center[0] + config.raw_width // 10, config.raw_center[1]): (0, 0, 0, 0, 0, 1, 0, 0, 0),# mid right
            (config.raw_center[0] - config.raw_width // 10, config.raw_center[1] - config.raw_height // 10): (0, 0, 0, 0, 0, 0, 1, 0, 0),  # bot left
            (config.raw_center[0], config.raw_center[1] - config.raw_height // 10): (0, 0, 0, 0, 0, 0, 0, 1, 0),  # bot
            (config.raw_center[0] + config.raw_width // 10, config.raw_center[1] - config.raw_height // 10): (0, 0, 0, 0, 0, 0, 0, 0, 1)# bot right
        }

    def get_keys(self):
        keys = []
        for key in self.key_list:
            if  platform_name == 'Windows':
                if wapi.GetAsyncKeyState(ord(key)):
                    keys.append(key)
        return keys

    def get_mouse_position(self, vector, isRaw=False):
        if isRaw:
            positions = self.vector_to_positions_raw
        else:
            positions = self.vector_to_positions
        v = tuple(vector)
        return positions[v]

    def get_mouse_vector(self, mouse, isRaw=False):
        if isRaw:
            vectors = self.position_to_vector_raw
        else:
            vectors = self.position_to_vector

        min_dist = (np.inf, None)

        for mouse_pos in vectors.keys():
            d = dist.euclidean(mouse_pos, mouse)
            if d < min_dist[0]:
                min_dist = (d, vectors[mouse_pos])
        
        if min_dist[1] is None:
            return list((0, 0, 0, 0, 1, 0, 0, 0, 0))
        else:
            return list(min_dist[1])

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


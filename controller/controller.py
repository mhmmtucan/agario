import time
import pyautogui

import numpy as np

from mss import mss

from utils import Config
from utils import InputCheck
from utils import platform_name

from frame.frame_processor import processV2

from frame.Recorder import Recorder

from network.network import convnet

from scipy.spatial import distance as dist

class Controller:
    def __init__(self,queue):
        self.config = Config()
        self.model = convnet(self.config)
        self.model.load('agario-convnet.model')
        self.recorder = Recorder()
        self.queue = queue

    def start_playing(self):
        paused = True
        ic = InputCheck(self.config)

        sct = mss()

        base_color = []
        unix_keys = []
        first_time = True
        prev_areas = [0] * 5
        
        # wait 5 seconds to hide the terminal and start the game
        print("Open agar.io in 5 seconds")
        for i in range(5,0,-1):
            print(i)
            time.sleep(1)

        # main loop
        while True:
            if platform_name == 'Windows':
                keys = ic.get_keys()
            else:
                keys = []
                unix_keys = self.queue.get()

            if paused == False:
                image = np.array(sct.grab(monitor=self.config.roi), dtype='uint8')
                #self.recorder.Record([image, image, image, image])

                if first_time:
                    base_color = image[image.shape[0]//2,image.shape[1]//2]
                    first_time = False

                frame, area, base_color = processV2(image, base_color, self.config)

                past_areas = np.array(prev_areas).reshape(-1, 5)
                current_area = np.array([area]).reshape(-1, 1)

                best_prediction = (-np.inf, [0, 0, 0, 0, 1, 0, 0, 0, 0])

                for i in range(9):
                    mouse = [0] * 9
                    mouse[i] = 1
                    m = np.array(mouse).reshape(-1, 9)
                    s = np.array([0]).reshape(-1, 1)

                    current_frame = np.array(frame).reshape(-1, 90, 150, 1)

                    main_input = current_frame
                    side_input = np.concatenate([past_areas, current_area, m, s], axis=1).reshape(-1, 16)

                    prediction = self.model.predict([main_input, side_input])[0] # returns a list with 5 elements, showing the future areas

                    d = dist.euclidean(prev_areas, prediction) # find better solution to find the best direction to go
                    if d >= best_prediction[0]: # euclidean distance always returns positive number
                        best_prediction = (d, mouse)

                target_mouse_pos = ic.get_mouse_position(best_prediction[1])
                pyautogui.moveTo(target_mouse_pos)
                
                prev_areas.pop(0)
                prev_areas.append(area)
            else:
                print('waiting the user to start the game')

            if 'Q' in keys or 'q' in unix_keys:
                print('quit playing')
                break

            if 'P' in keys or 'p' in unix_keys: # pause
                print('pause the playing')
                paused = True
                base_color = []

            if 'C' in keys or 'c' in unix_keys: # continue
                print('continue playing')
                paused = False
                base_color = []
                first_time = True
                prev_areas = [0] * 5
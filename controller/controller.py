import queue

import numpy as np
import pyautogui
from mss import mss

from frame.Recorder import Recorder
from frame.frame_processor import process
from trainer.network import convnet
from utils import Config
from utils import InputCheck
from utils import platform_name
from keras.models import load_model
from sklearn import preprocessing

class Controller:
    def __init__(self,q,model_file):
        self.config = Config()
        #self.model = convnet(self.config)
        #self.model.load('agario-convnet.model')
        self.model = load_model(model_file)
        self.recorder = Recorder()
        self.q = q

    def start_playing(self):
        paused = True
        ic = InputCheck(self.config)

        sct = mss()
        min_max_scaler = preprocessing.MinMaxScaler()
        max_abs_scaler = preprocessing.MaxAbsScaler()

        base_color = []

        first_time = True
        prev_areas = [750] * 5
        
        # wait 5 seconds to hide the terminal and start the game
        print("Press c to start game")

        # main loop
        while True:
            unix_keys = []
            if platform_name == 'Windows':
                keys = ic.get_keys()
            else:
                keys = []
                try:
                    unix_keys = self.q.get(block=False)
                except queue.Empty:
                    pass

            if paused == False:
                image = np.array(sct.grab(monitor=self.config.roi), dtype='uint8')
                #self.recorder.Record([image, image, image, image])
                #self.recorder.Record(image, './gameplay/')

                if first_time:
                    base_color = image[image.shape[0]//2,image.shape[1]//2]
                    first_time = False

                resized_image, frame, area = process(image, base_color, self.config)
                #self.recorder.Record(resized_image, './gameplay/')
                past_areas = np.array(prev_areas)
                current_area = np.array([area])

                best_prediction = (-np.inf, [0, 0, 0, 0, 1, 0, 0, 0, 0])

                # convert data for model
                past_areas = np.concatenate((past_areas,[0,750]))
                past_areas = np.around(min_max_scaler.fit_transform(past_areas.reshape(-1, 1)).flatten()[:-2], decimals=3)
                current_area = np.concatenate((current_area, [0, 750]))
                current_area = np.around(min_max_scaler.fit_transform(current_area.reshape(-1, 1)).flatten()[:-2],
                                       decimals=3)

                frame = np.around(max_abs_scaler.fit_transform(frame),decimals=3)

                past_areas = past_areas.reshape(-1, 5)
                current_area = current_area.reshape(-1, 1)

                for i in range(9):
                    mouse = [0] * 9
                    mouse[i] = 1
                    m = np.array(mouse).reshape(-1, 9)
                    s = np.array([0]).reshape(-1, 1)

                    current_frame = np.array(frame).reshape(-1, 144, 256, 1)

                    main_input = current_frame
                    side_input = np.concatenate([past_areas, current_area, m, s], axis=1).reshape(-1, 16)

                    prediction = self.model.predict([main_input, side_input])[0] # returns a list with 5 elements, showing the future areas

                    #d = dist.euclidean(prev_areas, prediction) # find better solution to find the best direction to go
                    #diff = np.array(prediction)-np.array(prev_areas)
                    diff = np.array(prediction)
                    weights = np.array([1,2,3,4,5])
                    d = np.dot(diff,weights)
                    #print(mouse,d)
                    if d >= best_prediction[0]: # euclidean distance always returns positive number
                        best_prediction = (d, mouse)

                target_mouse_pos = ic.get_mouse_position(best_prediction[1])
                pyautogui.moveTo(target_mouse_pos)
                #pyautogui.click(target_mouse_pos)
                
                prev_areas.pop(0)
                prev_areas.append(area)

            if 'Q' in keys or 'q' in unix_keys:
                print('quit playing')
                break

            if 'P' in keys or 'p' in unix_keys: # pause
                print('pause the playing')
                paused = True

            if 'C' in keys or 'c' in unix_keys: # continue
                print('continue playing')
                paused = False
                base_color = []
                first_time = True
                prev_areas = [750] * 5
import time
import pyautogui

import numpy as np

from mss import mss
from utils import Config
from utils import InputCheck
from utils import ExperienceBuffer
from frame_processor import process

if __name__ == '__main__':
    config = Config()

    paused = False
    window_width, window_height = pyautogui.size()
    ic = InputCheck(config)

    max_buffer_size = 50000
    session_buffer = ExperienceBuffer(max_buffer_size)

    filenames = ['first', 'second', 'third', 'fourth', 'fifth']
    filename_index = 0

    sct = mss()
    
    areas = [] # every element will be a list with 5 elements
    frames = [] # every element will be the processed version of the screen
    mouses = [] # every element will be a tuple with 2 elements -> we can change this in to 1 x 8 one hot key vector
    spaces= [] # every element will be an binary integer 0 or 1

    base_color = []

    # wait 5 seconds to hide the terminal and start the game
    print("Open agar.io in 5 seconds")
    for i in range(5,0,-1):
        print(i)
        time.sleep(1)

    # main loop
    while True:
        keys = ic.get_keys()

        if paused == False:
            image = np.array(sct.grab(monitor=config.roi), dtype='uint8')

            frame, area, base_color = process(image, base_color, config)

            areas.append(area)
            frames.append(frame)
            mouses.append(ic.get_mouse_vector(pyautogui.position()))
            # should space elemnts be 1x1 lists or normal integer
            if ' ' in keys:
                spaces.append(1)
            else:
                spaces.append(0)

            # also get the space key hits

        if 'Q' in keys:
            print('quit processing')
            break

        if 'P' in keys: # pause            
            print('pause the processing')
            paused = True
            base_color = []

            print('processing the user experience started')
            # create a function instead of this
            frame_num = len(frames)
            look_up = 5

            # we might want to remove last five frame when processing
            # if we remove last five frame from the frames list we can have non-zero last areas
            # if we dont we can remove the frames after the character died so we could get rid of menu frames
            for i, frame in enumerate(frames):
                past_areas = []
                area = areas[i]
                future_areas = []
                frame = frames[i]
                mouse = mouses[i]
                space = spaces[i]

                if i < look_up:
                    past_areas = [0] * (look_up - i) + areas[0 : i]
                else:
                    past_areas = areas[i - look_up : i]

                if i >= frame_num - look_up:
                    future_areas = areas[i + 1 : frame_num] + [0] * (i + look_up - frame_num + 1)
                else:
                    future_areas = areas[i + 1 : i + 6]


                #print('frame number: {}, past_areas: {}'.format(i, past_areas))
                #print('frame number: {}, area: {}'.format(i, area))
                #print('frame number: {}, future_areas: {}'.format(i, future_areas))
                #print('frame number: {}, frame: \n{}'.format(i, frame))
                #print('frame number: {}, mouse: {}'.format(i, mouse))
                #print('frame number: {}, space: {}'.format(i, space))
                #print('')

                # need to change the name of the file everytime or need to do load and save which will take more time
                # used half of the screen as roi and resized to half when processing
                # 75 frame took 20mb
                session_buffer.add(np.reshape(np.array([past_areas, area, future_areas, mouse, space, frame]), [1, 6]))
                session_buffer.save('training_data.npy')

                if session_buffer.length == max_buffer_size:
                    session_buffer = ExperienceBuffer(max_buffer_size)
                    print('experience buffer is full')

            print('processing the user experience finnished')

        if 'C' in keys: # continue
            print('continue processing')
            paused = False
            base_color = []

            # restart everything
            areas = [] # every element will be a list with 5 elements
            frames = [] # every element will be the processed version of the screen
            mouse_positions = [] # every element will be a tuple with 2 elements -> we can change this in to 1 x 8 one hot key vector
            space_hits = [] # every element will be an binary integer 0 or 1
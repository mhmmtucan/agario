import os
import uuid
import cv2
import queue
import pyautogui

import numpy as np

from mss import mss

from utils import say
from utils import Config
from utils import InputCheck
from utils import platform_name
from utils import ExperienceBuffer

from frame.frame_processor import processV2
from frame.Recorder import Recorder

def combine_data(foldername, outfilename):
    combined_data = []
    for file in os.listdir(foldername):
        if file == '.DS_Store':
            continue
        new_data = np.load(foldername + file)
        combined_data.extend(new_data)
    
    np.save(outfilename, np.reshape(np.array(combined_data), [-1, 6]))

def fix_data(filename):
    experience_buffer = ExperienceBuffer()
    print('loading data to fix')
    data = np.load(filename)
    for d in data:
        past_areas = d[0]
        current_area = d[1]
        future_areas = d[2]
        mouse = d[3]
        space = d[4]
        current_frame = d[5]

        past_areas = np.array(past_areas).reshape(5)
        current_area = np.array([current_area]).reshape(1)
        future_areas = np.array(future_areas).reshape(5)
        mouse = np.array(mouse).reshape(9)
        space = np.array([space]).reshape(1)

        experience_buffer.add(np.reshape(np.array([past_areas, current_area, future_areas, mouse, space, current_frame]), [1, 6]))

    experience_buffer.save(filename)
    print('saved the fixed data')

def convert_data(areas, frames, mouses, spaces):
    episode_buffer = ExperienceBuffer()

    frame_num = len(frames)
    look_up = 5

    # we might want to remove last five frame when processing
    # if we remove last five frame from the frames list we can have non-zero last areas
    # if we dont we can remove the frames after the character died so we could get rid of menu frames
    for i, frame in enumerate(frames):
        past_areas = []
        current_area = areas[i]
        future_areas = []
        current_frame = frames[i]
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

        past_areas = np.array(past_areas)
        current_area = np.array([current_area])
        future_areas = np.array(future_areas)
        mouse = np.array(mouse)
        space = np.array([space])

        episode_buffer.add(np.reshape(np.array([past_areas, current_area, future_areas, mouse, space, current_frame]), [1, 6]))

    return episode_buffer

def start_collecting(filename, q, isRaw):
    #fix_data('training_data.npy')

    config = Config()

    paused = True
    ic = InputCheck(config)

    max_buffer_size = 50000
    session_buffer = ExperienceBuffer(max_buffer_size)

    sct = mss()

    areas = [] # every element will be a list with 5 elements
    frames = [] # every element will be the processed version of the screen
    mouses = [] # every element will be a tuple with 2 elements -> we can change this in to 1 x 8 one hot key vector
    spaces= [] # every element will be an binary integer 0 or 1

    base_color = []
    unix_keys = []
    first_time = True
    print("Game started")
    say("game started")
    # main loop
    while True:
        unix_keys = []
        if platform_name == 'Windows':
            keys = ic.get_keys()
        else:
            keys = []
            try:
                unix_keys = q.get(block=False)
            except queue.Empty:
                pass

        if paused == False:
            image = np.array(sct.grab(monitor=config.roi), dtype='uint8')

            if first_time:
                base_color = image[image.shape[0]//2,image.shape[1]//2]
                first_time = False

            if isRaw:
                raw_image = cv2.resize(image,(1280,720))
                recorder.Record(raw_image, filename + "_f")
            else:
                resized_image, frame, area, base_color = processV2(image, base_color, config)
                # TODO: change area to radius

                #cv2.imshow('frame', frame)
                #cv2.moveWindow('frame', 0, 0)
                #cv2.waitKey(1)
                areas.append(area)
                frames.append(frame)

                recorder.Record(resized_image, filename + "_f")

            mouses.append(ic.get_mouse_vector(pyautogui.position()))
            # should space elemnts be 1x1 lists or normal integer
            if ' ' in keys or ' ' in unix_keys:
                spaces.append(1)
            else:
                spaces.append(0)

            # record game frame along with mouse and space for future uses
            # fix record size to 720p

        if 'Q' in keys or 'q' in unix_keys:
            print('quit processing')
            say("quit")
            break

        if 'S' in keys or 's' in unix_keys: # pause
            print('pause the processing')
            say("saved")
            paused = True

            base_color = []

            print('processing the user experience started')
            # create a function instead of this
            session_buffer.add(convert_data(areas, frames, mouses, spaces).buffer)
            session_buffer.save(filename+".npy")

            print('processing the user experience finnished')
            print('experience length: {}'.format(session_buffer.length()))

            if session_buffer.length() == max_buffer_size:
                session_buffer = ExperienceBuffer(max_buffer_size)
                print('experience buffer is full')
                say("buffer is full")

        if 'N' in keys or 'n' in unix_keys: # pause w/o saving
            print('processing paused - not saved')
            say("not saved")
            paused = True
            base_color = []

        if 'R' in keys or 'r' in unix_keys: # only raw data saved
            print('raw data saved')
            say("raw saved")

            np.save(filename+"_m.npy",mouses)
            np.save(filename+"_s.npy",spaces)

            foldername = './raw_data/'
            filename = foldername + str(uuid.uuid4())

            paused = True
            mouses = []
            spaces = []

        if 'C' in keys or 'c' in unix_keys: # continue
            print('continue processing')
            say("continue")
            paused = False
            base_color = []
            recorder = Recorder()

            # restart everything
            areas = [] # every element will be a list with 5 elements
            frames = [] # every element will be the processed version of the screen
            mouse_positions = [] # every element will be a tuple with 2 elements -> we can change this in to 1 x 8 one hot key vector
            space_hits = [] # every element will be an binary integer 0 or 1
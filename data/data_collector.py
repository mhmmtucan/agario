import os
import queue
import shutil
import uuid

import cv2
import numpy as np
import pyautogui
from mss import mss
from scipy import sparse

from frame.Recorder import Recorder
from frame.frame_processor import process
from utils import Config
from utils import ExperienceBuffer
from utils import InputCheck
from utils import platform_name
from utils import say


def convert_data(diameters, frames, mouses, spaces):
    episode_buffer = ExperienceBuffer()
    frame_num = len(frames)
    look_up = 5

    # we might want to remove last five frame when processing
    # if we remove last five frame from the frames list we can have non-zero last areas
    # if we dont we can remove the frames after the character died so we could get rid of menu frames
    for i, frame in enumerate(frames):
        past_diameter = []
        current_diameter = diameters[i]
        future_diameter = []
        current_frame = frames[i]
        mouse = mouses[i]
        space = spaces[i]

        if i < look_up:
            past_diameter = [750] * (look_up - i) + diameters[0: i]
        else:
            past_diameter = diameters[i - look_up: i]

        if i >= frame_num - look_up:
            future_diameter = diameters[i + 1: frame_num] + [0] * (i + look_up - frame_num + 1)
        else:
            future_diameter = diameters[i + 1: i + 6]

        past_diameter = np.array(past_diameter)
        current_diameter = np.array([current_diameter])
        future_diameter = np.array(future_diameter)
        mouse = np.array(mouse)
        space = np.array([space])

        episode_buffer.add(
            np.reshape(np.array([past_diameter, current_diameter, future_diameter, mouse, space, sparse.csc_matrix(current_frame)]),
                       [1, 6]))
        episode_buffer.add(
            np.reshape(np.array([past_diameter, current_diameter, future_diameter,
                                 np.fliplr(np.array(mouse).reshape(3, 3)).flatten().tolist(), space,
                                 sparse.csc_matrix(np.fliplr(np.array(current_frame)))]),
                       [1, 6]))
        episode_buffer.add(
            np.reshape(np.array([past_diameter, current_diameter, future_diameter,
                                 np.flipud(np.array(mouse).reshape(3, 3)).flatten().tolist(), space,
                                 sparse.csc_matrix(np.flipud(np.array(current_frame)))]),
                       [1, 6]))
        episode_buffer.add(
            np.reshape(np.array([past_diameter, current_diameter, future_diameter,
                                 np.flipud(np.fliplr(np.array(mouse).reshape(3, 3))).flatten().tolist(), space,
                                 sparse.csc_matrix(np.flipud(np.fliplr(np.array(current_frame))))]),
                       [1, 6]))

    return episode_buffer


def start_collecting(foldername, q, isRaw):
    # fix_data('training_data.npy')

    config = Config()
    ic = InputCheck(config)
    max_buffer_size = 50000
    session_buffer = ExperienceBuffer(max_buffer_size)
    sct = mss()

    base_color = []
    first_time = True
    paused = True

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

        if not paused:
            image = np.array(sct.grab(monitor=config.roi), dtype='uint8')

            if first_time:
                base_color = image[image.shape[0] // 2, image.shape[1] // 2]
                first_time = False

            if isRaw:
                raw_image = cv2.resize(image, (config.raw_width, config.raw_height))
                recorder.Record(raw_image, filename)
                # mouse positions adjusted according to raw data screen resolution
                mouses.append([int(i * config.raw_screen_ratio) for i in pyautogui.position()])
            else:
                resized_image, frame, diam = process(image, base_color, config)
                diameters.append(diam)
                frames.append(frame)
                recorder.Record(resized_image, filename)
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

        if 'S' in keys or 's' in unix_keys:  # pause
            print('pause the processing')
            say("saved")
            paused = True

            base_color = []

            print('processing the user experience started')
            session_buffer.add(convert_data(diameters, frames, mouses, spaces).buffer)
            session_buffer.save(filename + "train_data.npy")

            print('processing the user experience finnished')
            print('experience length: {}'.format(session_buffer.length()))

            if session_buffer.length() == max_buffer_size:
                session_buffer = ExperienceBuffer(max_buffer_size)
                print('experience buffer is full')
                say("buffer is full")

        if 'N' in keys or 'n' in unix_keys:  # pause w/o saving
            print('processing paused - not saved')
            say("not saved")
            paused = True
            # delete file since raw not saved
            shutil.rmtree(filename)

        if 'R' in keys or 'r' in unix_keys:  # only raw data saved
            print('raw data saved')
            say("raw saved")

            np.save(filename + "mouses.npy", mouses)
            np.save(filename + "spaces.npy", spaces)
            paused = True

        if 'C' in keys or 'c' in unix_keys:  # continue
            print('continue processing')
            say("continue")
            paused = False
            base_color = []
            recorder = Recorder()

            filename = foldername + str(uuid.uuid4())
            if not os.path.exists(filename):
                os.makedirs(filename)
            filename += '/'

            # restart everything
            diameters = []  # every element will be a list with 5 elements
            frames = []  # every element will be the processed version of the screen
            mouses = []  # every element will be a tuple with 2 elements -> we can change this in to 1 x 9 one hot key vector
            spaces = []  # every element will be an binary integer 0 or 1


# combine and process recorded raw data
# mouse positions adjusted according to raw data screen resolution, no need to think here
def combine_process_raw():
    config = Config()
    ic = InputCheck(config)
    foldername = "./raw_data/"
    used_data_folder = foldername + '_used/'
    combined_raw = []
    total_frame = 0
    chop_from_end = -15
    total_folder = len(os.listdir(foldername)) - 1
    folder_count = 0
    batch_count = 0

    if '.DS_Store' in os.listdir(foldername):
        os.remove(foldername + '/.DS_Store')
        total_folder -= 1

    for i, subfolder in enumerate(os.listdir(foldername)):
        if subfolder == '_used':
            continue

        print("\nLoading folder: ", subfolder)
        # fetch all required data
        cap = cv2.VideoCapture(foldername + subfolder + '/frames.avi')
        mouses = np.load(foldername + subfolder + '/mouses.npy')
        spaces = np.load(foldername + subfolder + '/spaces.npy')
        mouses = mouses[:chop_from_end]
        spaces = spaces[:chop_from_end]
        frames = []
        diameters = []

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count += chop_from_end

        # check_raw_data(frame_count,mouses,spaces)

        base_color = []
        first_time = True
        recorder = Recorder()

        # process image
        for j in range(int(frame_count)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            _, image = cap.read()

            if first_time:
                base_color = image[image.shape[0] // 2, image.shape[1] // 2]
                first_time = False

            resized_image, frame, diam = process(image, base_color, config)
            # apply sparse here,but can be problem with convert data function
            frames.append(frame)
            diameters.append(diam)

            #cv2.imshow("frame", resized_image)
            #recorder.Record(resized_image, foldername + subfolder + '/processed_')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        total_frame += frame_count

        processed_mouses = []
        # process mouses to mouse vector
        # mouses are raw data, which is x and y positions according to raw data res, so get_mouse_vector for raw resolution
        for m in mouses:
            processed_mouses.append(ic.get_mouse_vector(m))

        new_data = convert_data(diameters, frames, processed_mouses, spaces).buffer

        combined_raw.extend(new_data)

        print("{}/{} folder finished".format(i + 1, total_folder))
        print("{}/{} folder frame/total frame\n".format(frame_count * 4, total_frame * 4))
        cap.release()
        folder_count += 1

        # put used data to used folder in order to prevent duplicate data
        shutil.move(foldername + subfolder, used_data_folder)

        if len(combined_raw) > 50000:
            np.savez_compressed("train_data/train_data"+str(batch_count)+"_"+str(len(combined_raw))+".npz",data=combined_raw)
            batch_count += 1
            print("train_data"+str(batch_count)+"_"+str(len(combined_raw))+".npz saved")
            combined_raw = []
            
    if len(combined_raw) > 0:  
        np.savez_compressed("train_data/train_data" + str(batch_count) + "_" + str(len(combined_raw)) + ".npz", data=combined_raw)
        print("train_data"+str(batch_count)+"_"+str(len(combined_raw))+".npz saved")

    '''
        if len(combined_raw) > 10000 or folder_count == total_folder:
            batch_count += 1
            save_data = np.reshape(np.array(combined_raw), [-1, 6])

            np.savez_compressed("train_data/train_data"+str(batch_count)+".npz", data=save_data)
            print("train_data"+str(batch_count)+".npz shape", save_data.shape)

            combined_raw = []
    '''
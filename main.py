# try to find better names for packages and modules

import uuid
import threading

from queue import Queue

from utils import Config
from utils import InputCheck
from utils import get_unix_keys
from utils import platform_name
from utils import ExperienceBuffer

from data.check_data import print_data
from data.check_data import print_play
from data.check_data import print_size

from data.data_collector import fix_data
from data.data_collector import combine_data
from data.data_collector import convert_data
from data.data_collector import start_collecting

from frame.frame_processor import processV2

from frame.Recorder import Recorder

from controller.controller import Controller

from network.network import convnet
from network.network import create_model
from network.network import train_convnet

foldername = './training-data/'
outfilename = 'training-data.npy'

def collectRaw(queue=None):
    foldername = './raw_data/'
    filename = foldername + str(uuid.uuid4())
    start_collecting(filename, queue, True)

def collectLive(queue=None):
    foldername = './videos/'
    filename = foldername + str(uuid.uuid4())
    start_collecting(filename, queue, False)

def combine():
    combine_data(foldername, outfilename)

def train():
    create_model(outfilename)
    #print_data(outfilename)
    #print_size(outfilename)
    #print_play(outfilename)

def control(queue=None):
    controller = Controller(queue)
    controller.start_playing()

if __name__ == '__main__':
    one_hot = [0,1,0,0,0]
    a, b, c, d, e = [one_hot[i] == 1 for i in range(5)]

    if platform_name != 'Windows':
        if d: train()
        elif c: combine()
        else:
            queue = Queue()
            if a:
                main_thread = threading.Thread(target=collectRaw, args=(queue,))
                main_thread.start()
            if b:
                main_thread = threading.Thread(target=collectLive, args=(queue,))
                main_thread.start()
            elif e:
                main_thread = threading.Thread(target=control, args=(queue,))
                main_thread.start()

            get_unix_keys(queue)
    else:
        if a: collectRaw()
        elif b: collectLive()
        elif c: combine()
        elif d: train()
        elif e: control()

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

from frame.frame_processor import process
from frame.frame_processor import processV2

from frame.Recorder import Recorder

from controller.controller import Controller

from network.network import convnet
from network.network import create_model
from network.network import train_convnet

def collect(queue=None):
    foldername = './training-data/'
    outfilename = 'training-data.npy'

    filename = foldername + str(uuid.uuid4()) + '.npy'
    start_collecting(filename, queue)
    
    combine_data(foldername, outfilename)

def train():
    training_file = 'training-data.npy'
    create_model(training_file)

    #print_data(training_file)
    #print_size(training_file)
    #print_play(training_file)

def control(queue=None):
    controller = Controller(queue)
    controller.start_playing()

if __name__ == '__main__':
    collect_data = True
    train_net = False
    test_controller = False

    if platform_name != 'Windows':
        if train_net:
            train()
        else:
            queue = Queue()
            if collect_data:
                main_thread = threading.Thread(target=collect, args=(queue,))
                main_thread.start()
            elif test_controller:
                main_thread = threading.Thread(target=control, args=(queue,))
                main_thread.start()

            get_unix_keys(queue)
    else:
        if collect_data: collect()
        elif train_net: train()
        elif test_controller: control()

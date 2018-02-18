# try to find better names for packages and modules

import threading
from queue import Queue

from utils import Config
from utils import InputCheck
from utils import ExperienceBuffer
from utils import platform_name
from utils import get_unix_keys

from data.check_data import print_data
from data.check_data import print_play
from data.check_data import print_size

from data.data_collector import fix_data
from data.data_collector import convert_data
from data.data_collector import start_collecting

from frame.frame_processor import process
from frame.frame_processor import processV2

from frame.Recorder import Recorder

from network.network import convnet
from network.network import create_model
from network.network import train_convnet

def main(queue=None):
    # use unique string to get new file name everytime
    # otherwise the datas fro previous sessions will be lost

    filename = 'trained-data/data.npy'

    start_collecting(filename,queue)
    #print_data(filename)
    #create_model(filename)


if __name__ == '__main__':
    if platform_name != 'Windows':
        queue = Queue()
        main_thread = threading.Thread(target=main, args=(queue,))
        main_thread.start()
        get_unix_keys(queue)
    else:
        main()
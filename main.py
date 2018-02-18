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

from network.network import convnet
from network.network import create_model
from network.network import train_convnet

def main(queue=None):
    foldername = './trained-data/'
    outfilename = 'trained_data.npy'

    collect_data = False

    if collect_data:
        # use unique string to get new file name everytime
        # otherwise the datas fro previous sessions will be lost
        filename = foldername + str(uuid.uuid4()) + '.npy'
        #filename = 'trained-data/data.npy'
        start_collecting(filename, queue)

    #combine_data(foldername, outfilename)
    #print_data(outfilename)
    #print_size(outfilename)
    #print_play(outfilename)
    create_model(outfilename)

if __name__ == '__main__':
    if platform_name != 'Windows':
        queue = Queue()
        main_thread = threading.Thread(target=main, args=(queue,))
        main_thread.start()
        get_unix_keys(queue)
    else:
        main()
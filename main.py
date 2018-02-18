# try to find better names for packages and modules

from utils import Config
from utils import InputCheck
from utils import ExperienceBuffer

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

if __name__ == '__main__':
    # use unique string to get new file name everytime
    # otherwise the datas fro previous sessions will be lost

    filename = 'trained-data/data.npy'

    #start_collecting(filename)
    #print_data(filename)
    create_model(filename)
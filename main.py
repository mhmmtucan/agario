# try to find better names for packages and modules
import threading
import uuid
from queue import Queue

from controller.controller import Controller
from data.data_collector import combine_data
from data.data_collector import combine_raw
from data.data_collector import start_collecting
from trainer.network import create_model
from utils import get_unix_keys
from utils import platform_name

def collectRaw(queue=None):
    foldername = './raw_data/'
    start_collecting(foldername, queue, True)

def collectLive(queue=None):
    foldername = './videos/'
    filename = foldername + str(uuid.uuid4())
    start_collecting(filename, queue, False)

def combine():
    foldername = './training-data/'
    outfilename = 'training-data.npy'
    combine_data(foldername, outfilename)

def train():
    processed_data = "combined_raw.npy"
    create_model(processed_data)
    #print_data(processed_data)
    #print_size(processed_data)
    #print_play(processed_data)

def control(queue=None):
    model_file='model.h5'
    controller = Controller(queue,model_file=model_file)
    controller.start_playing()

if __name__ == '__main__':
    # collect raw, combine raw, collect live, combine, train, control
    one_hot = [0,0,0,0,0,1]
    a, b, c, d, e, f = [one_hot[i] == 1 for i in range(6)]

    if platform_name != 'Windows':
        if b: combine_raw()
        elif d: combine()
        elif e: train()
        else:
            queue = Queue()
            if a:
                main_thread = threading.Thread(target=collectRaw, args=(queue,))
                main_thread.start()
            elif c:
                main_thread = threading.Thread(target=collectLive, args=(queue,))
                main_thread.start()
            elif f:
                main_thread = threading.Thread(target=control, args=(queue,))
                main_thread.start()

            get_unix_keys(queue)
    else:
        if a: collectRaw()
        elif b: combine_raw()
        elif c: collectLive()
        elif d: combine()
        elif e: train()
        elif f: control()
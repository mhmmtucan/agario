# try to find better names for packages and modules
import threading
from queue import Queue

from controller.controller import Controller
from data.data_collector import combine_raw
from data.data_collector import start_collecting
from trainer.network import create_model
from utils import get_unix_keys
from utils import platform_name

def collectRaw(queue=None):
    isRaw = True
    if isRaw:
        foldername = './raw_data/'
        start_collecting(foldername, queue, True)
    else:
        foldername = './tmp/'
        start_collecting(foldername, queue, False)

def train():
    processed_data = "train_data.npy"
    create_model(processed_data)
    #print_data(processed_data)
    #print_size(processed_data)
    #print_play(processed_data)

def control(queue=None):
    model_file='model.h5'
    controller = Controller(queue,model_file=model_file)
    controller.start_playing()

if __name__ == '__main__':
    # collect raw, combine raw, train, control
    one_hot = [0,1,0,0]
    a, b, c, d = [one_hot[i] == 1 for i in range(6)]

    if platform_name != 'Windows':
        if b: combine_raw()
        elif c: train()
        else:
            queue = Queue()
            if a:
                main_thread = threading.Thread(target=collectRaw, args=(queue,))
                main_thread.start()
            elif d:
                main_thread = threading.Thread(target=control, args=(queue,))
                main_thread.start()

            get_unix_keys(queue)
    else:
        if a: collectRaw()
        elif b: combine_raw()
        elif c: train()
        elif d: control()
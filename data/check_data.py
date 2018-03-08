from collections import Counter

import numpy as np
import pandas as pd


def print_data(filename):
    data = np.load(filename)

    df = pd.DataFrame(data)

    print('len')
    print(len(data)) # printing how much data we got so far
    print('')

    print('first ten data')
    print(df.head(10)) # printing the first ten data
    print('')

    print('last ten data')
    print(df.tail(10)) # printing the last ten data
    print('')

    print('mouse position counter')
    print(Counter(df[3].apply(str))) # printing the count of the mouse position -> 3 = mouse position
    print('')

    print('space key counter')
    print(Counter(df[4].apply(str))) # printing the count of the space key strokes -> 4 = space key
    print('')

def print_size(filename):
    data = np.load(filename)
    
    prevs = data[:, 0]
    currs = data[:, 1]
    futus = data[:, 2]
    mouss = data[:, 3]
    spacs = data[:, 4]
    frams = data[:, 5]

    print('prev size: {}, cur size: {}, fut size: {}, mous size: {}, space size: {}, frame size: {}'.format(prevs.shape, currs.shape, futus.shape, mouss.shape, spacs.shape, frams.shape))
    print('prev dtype: {}, cur dtype: {}, fut dtype: {}, mous dtype: {}, space dtype: {}, frame dtype: {}'.format(prevs.dtype, currs.dtype, futus.dtype, mouss.dtype, spacs.dtype, frams.dtype))
    print('')
    print('prev size: {}, cur size: {}, fut size: {}, mous size: {}, space size: {}, frame size: {}'.format(prevs[0].shape, currs[0].shape, futus[0].shape, mouss[0].shape, spacs[0].shape, frams[0].shape))
    print('prev dtype: {}, cur dtype: {}, fut dtype: {}, mous dtype: {}, space dtype: {}, frame dtype: {}'.format(prevs[0].dtype, currs[0].dtype, futus[0].dtype, mouss[0].dtype, spacs[0].dtype, frams[0].dtype))

def print_play(filename):
    data = np.load(filename)
    
    prevs = np.array([i[0] for i in data], dtype=np.float32).reshape((-1, 5))
    currs = np.array([i[1] for i in data], dtype=np.float32).reshape((-1, 1))
    futus = np.array([i[2] for i in data], dtype=np.float32).reshape((-1, 5))
    mouss = np.array([i[3] for i in data], dtype=np.float32).reshape((-1, 9))
    spacs = np.array([i[4] for i in data], dtype=np.float32).reshape((-1, 1))
    frams = np.array([i[5] for i in data], dtype=np.float32).reshape((-1, 90, 150, 1))
    
    print('prev size: {}, cur size: {}, fut size: {}, mous size: {}, space size: {}, frame size: {}'.format(prevs.shape, currs.shape, futus.shape, mouss.shape, spacs.shape, frams.shape))
    print('prev dtype: {}, cur dtype: {}, fut dtype: {}, mous dtype: {}, space dtype: {}, frame dtype: {}'.format(prevs.dtype, currs.dtype, futus.dtype, mouss.dtype, spacs.dtype, frams.dtype))
    print('')
    print('prev size: {}, cur size: {}, fut size: {}, mous size: {}, space size: {}, frame size: {}'.format(prevs[0].shape, currs[0].shape, futus[0].shape, mouss[0].shape, spacs[0].shape, frams[0].shape))
    print('prev dtype: {}, cur dtype: {}, fut dtype: {}, mous dtype: {}, space dtype: {}, frame dtype: {}'.format(prevs[0].dtype, currs[0].dtype, futus[0].dtype, mouss[0].dtype, spacs[0].dtype, frams[0].dtype))

def check_raw_data(frame_count,mouses,spaces):
    print('frame count: {} - mouses count {} - spaces count {}'.format(int(frame_count),len(mouses),len(spaces)))
    print(Counter(spaces))

    print("Mouses shape ",mouses.shape)
    print("Spaces shape ", spaces.shape)

    df_m = pd.DataFrame(mouses)
    print("Mouses head\n", df_m.head(10))
    print("Mouses tail\n", df_m.tail(10))

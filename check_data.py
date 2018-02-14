import numpy as np
import pandas as pd

from collections import Counter

data = np.load('training_data.npy')

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
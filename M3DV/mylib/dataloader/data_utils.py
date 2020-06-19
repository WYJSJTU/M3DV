import numpy as np
import os
import pandas as pd
import torch
def random_shift(center, move):
    offset = np.random.randint(-move, move + 1, size=3)
    shift_center = np.array(center) // 2 + offset
    return shift_center
def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped


def rotate(array,angle):
    '''
    @auther duducheng'''

    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z
def reflection(array,axis):
    if axis!=-1:
        array=np.flip(array,axis)
    else :
        array=np.copy(array)
    return array

def generate_test_csv(path,predicted=None):
    listName = []
    for fileName in os.listdir(os.path.join(path,'test')):
        if os.path.splitext(fileName)[1] == '.npz':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    index = list(range(len(listName)))
    val_frame = pd.DataFrame({'name': listName, 'predicted':predicted})
    val_frame.to_csv(os.path.join(path,"test.csv"), index=False)
    print(listName)

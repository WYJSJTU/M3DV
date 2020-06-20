import numpy as np

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

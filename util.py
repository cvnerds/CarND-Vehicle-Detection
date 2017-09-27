import cv2
import matplotlib.image as mpimg
import numpy as np
import os
from matplotlib import cm



# utility function that traverses a folder recursively
def walk_r(path):
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(path):
        for file in files:
            yield os.path.join(root, file)



def read_images(imgs):
    N = len(imgs)
    image_list = [0] * N
    for i,file in enumerate(imgs):
        image = mpimg.imread(file)
        # only for mpimg and png images, normalise back to [0,255]
        if file.endswith('.png'):
            image = image.astype(np.float32)*255
        image_list[i] = image
    return image_list



def filter_ext(files, ext):
    return filter(lambda path: path.endswith(ext), files)



def convert_color(img, conv='RGB2YCrCb'):
    if conv=='RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv=='RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv=='RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv=='RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv=='RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return np.copy(img)



import matplotlib.pyplot as plt

def plotList(images, titles=[], cmaps=[], shape=None, figsize=None, plot_axis='on'):
    
    N = len(images)
    
    if shape is None:
        shape = (1,N)
    
    rows = shape[0]
    cols = shape[1]
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    
    for row in range(rows):
        for col in range(cols):
            
            ax = axes[row,col] if rows > 1 and cols > 1 else axes[col]

            cmap = None

            i = row * cols + col
            if i < N:
                if i < len(cmaps):
                    cmap = cmaps[i]
                ax.imshow(images[i].astype(np.uint8), cmap=cmap)
                ax.axis(plot_axis)
                if i < len(titles):
                    ax.title.set_text(titles[i])
            else:
                ax.axis('off')    
    return fig
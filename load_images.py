import numpy as np
from PIL import Image
import os
def load_data(data_dir, classes, image_size, samples):
    paths = []
    images = []
    labels = []
    image_size = (image_size, image_size)
    for i in range(classes):
        path = os.path.join(data_dir, str(i))
        imglist = os.listdir(path)
        for sample in range(samples // classes):
            img = Image.open(os.path.join(path, imglist[sample])).convert('L')
            img = img.resize(image_size)
            img_array = np.array(img, dtype='float32') / 255.0
            paths.append(os.path.join(path, imglist[sample]))
            images.append(img_array)
            labels.append(int(i))
    return np.array(paths), np.array(images), np.array(labels)
'''
#----------load_images.py test----------#
data_dir = './train'
classes = 5
image_size = 28
samples = 10
images, labels = load_data(data_dir, classes, image_size, samples)
'''
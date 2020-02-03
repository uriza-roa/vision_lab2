from os import path
from skimage.io import imread
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from code import *

cwd = path.abspath('')
mafe = img_as_float(imread(path.join(cwd,'img','mafe_bebe.jpeg')))
andres = img_as_float(imread(path.join(cwd,'img','andres_bebe.jpg')))
blended_im = blended_image(mafe,andres,9,1)
plt.imshow(blended_im)
plt.axis('off')
plt.show()

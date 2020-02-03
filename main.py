from os import path
from skimage.io import imread
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from code import *

cwd = path.abspath('')
mafe = img_as_float(imread(path.join(cwd,'img','mafe_bebe.jpg')))
andres = img_as_float(imread(path.join(cwd,'img','andres_bebe.jpg')))
plt.imshow(andres)
plt.axis('off')
plt.show()


hybrid_im = hybrid_image(mafe,andres,4)
plt.imshow(hybrid_im)
plt.axis('off')
plt.show()

blended_im = blended_image(mafe,andres,9,1)
plt.imshow(blended_im)
plt.axis('off')
plt.show()

from os import path
from skimage.io import imread
import matplotlib.pyplot as plt
from code import *

cwd = path.abspath('')
image = imread(path.join(cwd,'img','mafdres.jpg'))
blended_image = blended_image(image,5)
plt.imshow(blended_image)
plt.axis('off')
plt.show()

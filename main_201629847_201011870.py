from os import path
from skimage.io import imread,imsave
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from code import *

#Image reading
cwd = path.abspath('')
im_mafe = img_as_float(imread(path.join(cwd,'img','mafe_bebe.jpg')))
im_andres = img_as_float(imread(path.join(cwd,'img','andres_bebe.jpg')))

#Creation of a hybrid image
hybrid_im = hybrid_image(im_mafe,im_andres,sigma=5)
imsave(path.join('results','hybrid.png'),hybrid_im)
fig = plt.figure()
plt.imshow(hybrid_im)
plt.draw()
plt.axis('off')
plt.pause(1)
input("<Hit Enter To Close>")
plt.close(fig)

im = hybrid_im.copy()
for i in range(6):
    im = pyramid_down(im,sigma=1)
    imsave(path.join('results',f'pyramid_{i+1}.png'),im)

#Creation of a blended image
blended_im = blended_image(im_mafe,im_andres,pyramid_size=9,sigma=1)
imsave(path.join('results','blended.png'),blended_im)
fig = plt.figure()
plt.imshow(blended_im)
plt.draw()
plt.axis('off')
plt.pause(1)
input("<Hit Enter To Close>")
plt.close(fig)

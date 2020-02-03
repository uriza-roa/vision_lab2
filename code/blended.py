import numpy as np
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.exposure import rescale_intensity

def pyramid_down(image,sigma=1):
    return rescale(gaussian(image,sigma,multichannel=True),0.5,multichannel=True)

def pyramid_up(image,sigma=1):
    return rescale(gaussian(image,sigma,multichannel=True),2,multichannel=True)

def gaussian_pyramid(image,pyramid_size,sigma=1):
    pyramid = []
    for i in range(pyramid_size):
        gaus = pyramid_down(image,sigma)
        pyramid.append(gaus)
        image = gaus
    return pyramid

def laplacian_pyramid(image,pyramid_size,sigma=1):
    gaus_pyramid = gaussian_pyramid(image,pyramid_size,sigma)
    gaus_pyramid.reverse()
    pyramid = []
    for i in range(len(gaus_pyramid)-1):
        lap = gaus_pyramid[i+1] - pyramid_up(gaus_pyramid[i],sigma)
        pyramid.insert(0,lap)
    lap = np.clip(image - pyramid_up(gaus_pyramid[-1],sigma),0,1)
    pyramid.insert(0,lap)
    return pyramid
    
def blended_image(image_1,image_2,pyramid_size,sigma=1):
    lapl_pyramid_1 = laplacian_pyramid(image_1,pyramid_size,sigma)
    lapl_pyramid_2 = laplacian_pyramid(image_2,pyramid_size,sigma)
    lapl_pyramid_1.reverse()
    lapl_pyramid_2.reverse()
    lapl_blended = []
    for l1,l2 in zip(lapl_pyramid_1,lapl_pyramid_2):
        hw = l1.shape[1]//2
        lb = np.hstack((l1[:,:hw],l2[:,hw:]))
        lapl_blended.append(lb)
    
    blended_image = lapl_blended[0]
    for i in range(1,pyramid_size):
        blended_image = pyramid_up(blended_image,sigma) + lapl_blended[i]
    blended_image = rescale_intensity(blended_image)
    return blended_image

# def blended_image(image,pyramid_size,sigma=1):
#     gaus_pyramid = gaussian_pyramid(image,pyramid_size,sigma)
#     lapl_pyramid = laplacian_pyramid(image,pyramid_size,sigma)
#     lapl_pyramid.reverse()
#     blended_image = gaus_pyramid[-1]
#     for laplacian in lapl_pyramid:
#         blended_image = pyramid_up(blended_image,sigma) + laplacian
#         blended_image = rescale_intensity(blended_image)
#     return blended_image
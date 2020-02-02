from skimage.filters import gaussian
from skimage.transform import rescale

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
    pyramid = []
    for i in range(pyramid_size):
        gaus = pyramid_down(image,sigma)
        lap = image - pyramid_up(gaus,sigma)
        if lap.min() < 0 :
            lap -= lap.min()
        pyramid.append(lap)
        image = gaus
    return pyramid

def blended_image(image,pyramid_size,sigma=1):
    gaus_pyramid = gaussian_pyramid(image,pyramid_size,sigma)
    lapl_pyramid = laplacian_pyramid(image,pyramid_size,sigma)
    blended_image = gaus_pyramid[-1]
    for laplacian in lapl_pyramid.reverse():
        blended_image = pyramid_up(blended_image,sigma) + laplacian
    return blended_image
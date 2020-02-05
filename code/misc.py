import numpy as np
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.exposure import rescale_intensity

def pyramid_down(image,sigma=1):
    """
    Applies a pyramid down operation to an image. First applies a Gaussian filter and then downsamples
    by a factor fo 2. 
    Args:
        image (numpy.ndarray): Image
        sigma (int, optional): Standard deviation of a Gaussian distribution. Defaults to 1.
    
    Returns:
        numpy.ndarray: Image after pyramid down
    """    
    return rescale(gaussian(image,sigma,multichannel=True),0.5,multichannel=True)

def pyramid_up(image,sigma=1):
    """
    Applies a pyramid up operation to an image. First applies a Gaussian filter and then upsamples
    by a factor fo 2. 
    Args:
        image (numpy.ndarray): Image
        sigma (int, optional): Standard deviation of a Gaussian distribution. Defaults to 1.
    
    Returns:
        numpy.ndarray: Image after pyramid up
    """ 
    return rescale(gaussian(image,sigma,multichannel=True),2,multichannel=True)

def gaussian_pyramid(image,pyramid_size,sigma=1):
    """
    Calculates the Gaussian pyramid of size pyramid_size from an input image.
    
    Args:
        image (numpy.ndarray): Image
        pyramid_size (int): Number of levels of the Gaussian pyramid
        sigma (int, optional): Standard deviation of a Gaussian distribution. Defaults to 1.
    
    Returns:
        list: A list containing the different levels of the Gaussian pyramid
    """    
    pyramid = []
    for i in range(pyramid_size):
        gaus = pyramid_down(image,sigma)
        pyramid.append(gaus)
        image = gaus
    return pyramid

def laplacian_pyramid(image,pyramid_size,sigma=1):
    """
    Calculates the Laplacian pyramid of size pyramid_size from an input image.
    
    Args:
        image (numpy.ndarray): Image
        pyramid_size (int): Number of levels of the Gaussian pyramid
        sigma (int, optional): Standard deviation of a Gaussian distribution. Defaults to 1.
    
    Returns:
        list: A list containing the different levels of the Laplacian pyramid
    """    
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
    """
    Creates a blended image from two input images by applying the Laplacian pyramid.
    The blended image consists of a union of the halves of the two input images.
    
    Args:
        image_1 (numpy.ndarray): Image to blend at the left of the resulting image
        image_2 (numpy.ndarray)): Image to blend at the right of the resulting image
        pyramid_size (int) Number of levels of the Gaussian pyramid
        sigma (int, optional): Standard deviation of a Gaussian distribution. Defaults to 1.
    
    Returns:
        numpy.ndarray: Blended image
    """    
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

def hybrid_image(image1,image2,sigma=1):
    """
    Creates a hybrid image from two input images by adding image1 after applying a lowpass filter
    and image2 after aplying a highpass filter.
    
    Args:
        image_1 (numpy.ndarray): Image to blend into the resulting hybrid image
        image_2 (numpy.ndarray)): Image to blend into the resulting hybrid image
        sigma (int, optional): Standard deviation of a Gaussian distribution. Defaults to 1.
    
    Returns:
        numpy.ndarray: Blended image
    """    
    low_pass = gaussian(image1,sigma)
    high_pass = np.clip(image2 - gaussian(image2,sigma),0,1)
    hybrid_image = np.clip(low_pass + high_pass,0,1)

    return hybrid_image
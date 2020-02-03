import numpy as np
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.exposure import rescale_intensity

def hybrid_image(image1,image2,sigma=1):
    low_pass = gaussian(image1,sigma)
    high_pass = np.clip(image2 - gaussian(image2,sigma),0,1)
    hybrid_image = np.clip(low_pass + high_pass,0,1)

    return hybrid_image
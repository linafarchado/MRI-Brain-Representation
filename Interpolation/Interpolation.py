import nibabel as nib
import numpy as np
import os
import random
import sys

class ImageInterpolator:
    def __init__(self, model):
        self.model = model
    
    def interpolate(self, image1, image2, alpha=random.uniform(0, 1)):
        # Encode the images
        latent1 = self.model.encode(image1)
        latent2 = self.model.encode(image2)
        
        # Interpolation
        interpolated_latent = alpha * latent1 + (1 - alpha) * latent2

        # Decode the interpolated latent
        interpolated_image = self.model.decode(interpolated_latent)
        
        return interpolated_image